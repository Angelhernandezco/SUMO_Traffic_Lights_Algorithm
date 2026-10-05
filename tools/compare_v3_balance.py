"""Evaluate a V3 variant and retain compact, paired comparison evidence."""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from run_sync import TRACE_NAMES
from sync.common import ROOT, DEMAND, END
from sync.metrics import write_csv
from sync.simulation import run
from tests.collect_validation import compact
from tests.validate_sync import canonical, compare, protected
from tools.compare_corridor import physical_coverage

LABELS = {"plain_offset": "Plain con offsets", "v1": "V1 — inicio",
          "v2": "V2 — copia", "v3": "V3 — ventana",
          "v3.1": "V3.1 — recortes según colas J2",
          "v3.2": "V3.2 — colas + cierre causal J2",
          "v3.3": "V3.3 — colas + recorte diferido J2",
          "v3.4": "V3.4 — recortes según colas del corredor",
          "v3.5": "V3.5 — colas + apertura alineada",
          "v3.6": "V3.6 — colas + margen de llegada",
          "v3.7": "V3.7 — colas + margen de llegada 2 s"}
REPORT_DIR = ROOT / "analysis/v3_balance"
RUNS_DIR = ROOT / "results/v3_balance"


def load_csv(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = json.loads(value)
                except (ValueError, TypeError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def evidence(result):
    cuts, extensions = {}, {}
    for event in result[3]:
        for cut in event.get("green_cuts", []):
            if isinstance(cut.get("applied_at"), (int, float)):
                key = f"{event['tls']}/phase{cut['phase']}"
                cuts[key] = cuts.get(key, 0) + cut["seconds_cut"]
                extensions[key] = extensions.get(key, 0) + cut.get("seconds_added", 0)
    return {"receivers": result[0]["receivers"], "cut_seconds_by_phase": cuts,
            "extra_noncorridor_green_by_phase": extensions,
            "early_corridor_closures": sum("corridor_early_close_at" in e for e in result[3]),
            "unused_corridor_seconds_removed": sum(e.get("unused_corridor_seconds_removed", 0) for e in result[3]),
            "deferred_notices": sum("deferred_at" in e for e in result[3]),
            "empty_deferred_notices_without_cuts": sum("deferred_at" in e and not e["crossings"] and
                not any(isinstance(c.get("applied_at"), (int, float)) and c["seconds_cut"] > 0
                        for c in e["green_cuts"]) for e in result[3]),
            "physical_windows": physical_coverage(result),
            "canonical_traces": {name: canonical(rows) for name, rows in zip(TRACE_NAMES, result[1:])}}


def collect():
    rows = [r for r in load_csv(ROOT / "analysis/six_configurations/per_demand.csv")
            if r["mode"] in ("plain_offset", "v1", "v2", "v3")]
    saved = REPORT_DIR / "per_demand.csv"
    records = {(r["mode"], r["seed"]): r for r in rows}
    if saved.exists():
        records.update({(r["mode"], r["seed"]): r for r in load_csv(saved) if r["mode"].startswith("v3.")})
    for path in sorted(RUNS_DIR.glob("seed*_v3.*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = compact(data["summary"])
        row["reference_equal"] = None  # New experiment, not a frozen replay.
        for tls, metrics in data["validation"]["physical_windows"].items():
            row[f"physical_window_coverage_{tls}_pct"] = metrics["coverage_pct"]
        records[(row["mode"], row["seed"])] = row
    return list(records.values())


def expected_cohort(seed):
    """Same full-corridor definition as FullObserver, including a J16 exit."""
    ids = set()
    for vehicle in ET.parse(DEMAND / f"seed_{seed}.rou.xml").iter("vehicle"):
        if float(vehicle.get("depart")) > END - 180:
            continue
        edges = vehicle.find("route").get("edges").split()
        if any(edges[i:i+4] == ["-E0", "E1", "E5", "E10"] for i in range(len(edges)-4)):
            ids.add(vehicle.get("id"))
    return sorted(ids)


def publish():
    rows = collect()
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    aggregates = {}
    lines = ["# V3: equilibrio entre progresión y espera", "",
             "Demandas 42–46, SUMO seed 42, 3600 s, paso 1 s. PPO oficial determinista sólo en J0. "
             "Offsets 0/0/72/24; viajes acumulados V3 12/22/32 s. Bases reutilizadas de la comparación validada de seis configuraciones.", "",
             "Objetivo práctico: al menos 84/93 sin parada, tiempo medio ≤38 s, waiting global <48495 y "
             "secundario <30041 veh·s. Umbrales de comparación, no garantías del controlador.", "",
             "| Versión | Corridas | Sin parada | Tiempo J0→J16 | Waiting E1 | Waiting corredor | Waiting secundario | Waiting red ± DE | Pendientes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in modes:
        group = [r for r in rows if r["mode"] == mode]
        n = len(group)
        total = sum(r["cohort_full"] for r in group)
        passed = sum(r["no_stop_full"] for r in group)
        means = {key: statistics.mean(r[key] for r in group) for key in
                 ("mean_J0_to_J16_s", "waiting_E1_s", "full_waiting_s", "secondary_total_s",
                  "network_waiting_s", "pending", "stops_per_vehicle", "waiting_time", "effective_flow", "avg_queue_length")}
        sd = statistics.stdev(r["network_waiting_s"] for r in group) if n > 1 else 0
        balanced = n == 5 and passed >= 84 and means["mean_J0_to_J16_s"] <= 38 and \
            means["network_waiting_s"] < 48495 and means["secondary_total_s"] < 30041
        aggregates[mode] = {"runs": n, "no_stop": passed, "cohort": total, "means": means,
                            "network_sd": sd, "meets_balance_target": balanced}
        lines.append(f"| {LABELS.get(mode, mode)} | {n} | {passed}/{total} ({100*passed/total:.2f}%) | "
                     f"{means['mean_J0_to_J16_s']:.2f} | {means['waiting_E1_s']:.1f} | {means['full_waiting_s']:.1f} | "
                     f"{means['secondary_total_s']:.1f} | {means['network_waiting_s']:.1f} ± {sd:.1f} | {means['pending']:.1f} |")
    lines += ["", "Todos los waiting son medias por corrida en veh·s. E1 usa los 232 vehículos rectos J0→J2; "
              "corredor completo usa 93 vehículos. Secundario suma aproximaciones no-corredor de J2/J10/J16, "
              "incluido el sentido contrario; red incluye todos los vehículos. El recorrido empieza al cruzar J0.", "",
              "## Resultados emparejados por demanda", "",
              "| Versión | Demanda | Sin parada | Paradas/veh. | Tiempo | Waiting E1 | Waiting secundario | Waiting red | Pendientes |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['seed']} | {r['no_stop_full']}/{r['cohort_full']} | "
                     f"{r['stops_per_vehicle']:.3f} | {r['mean_J0_to_J16_s']:.2f} | {r['waiting_E1_s']} | "
                     f"{r['secondary_total_s']} | {r['network_waiting_s']} | {r['pending']} |")
    finished = [mode for mode, a in aggregates.items() if mode.startswith("v3.") and a["runs"] == 5]
    balanced = [mode for mode in finished if aggregates[mode]["meets_balance_target"]]
    choices = balanced or [mode for mode in finished if aggregates[mode]["no_stop"] >= 79 and
                            aggregates[mode]["means"]["secondary_total_s"] < 30041 and
                            aggregates[mode]["means"]["network_waiting_s"] < 48495]
    recommended = min(choices, key=lambda mode: aggregates[mode]["means"]["network_waiting_s"]) if choices else None
    paired = {}
    if recommended:
        chosen = aggregates[recommended]
        lines += ["", "## Recomendación comparativa", "",
                  f"Candidata recomendada: **{recommended}**. " +
                  ("Cumple los cuatro umbrales prácticos de equilibrio." if balanced else
                   "Mejora el equilibrio frente a plain con offsets, pero todavía no cumple todos los umbrales de conservación de V3."), "",
                  "| Referencia | Diferencia sin parada | Cambio tiempo | Cambio waiting secundario | Cambio waiting red |",
                  "|---|---:|---:|---:|---:|"]
        for base in ("plain_offset", "v3"):
            b = aggregates[base]
            changes = {key: 100*(chosen["means"][key]/b["means"][key]-1) for key in
                       ("mean_J0_to_J16_s", "secondary_total_s", "network_waiting_s")}
            lines.append(f"| {LABELS[base]} | {chosen['no_stop']-b['no_stop']:+d} vehículos | "
                         f"{changes['mean_J0_to_J16_s']:+.2f}% | {changes['secondary_total_s']:+.2f}% | "
                         f"{changes['network_waiting_s']:+.2f}% |")
            pairs = [(r, next(other for other in rows if other["mode"] == base and other["seed"] == r["seed"]))
                     for r in rows if r["mode"] == recommended]
            paired[base] = {"no_stop_delta": chosen["no_stop"]-b["no_stop"], "mean_changes_pct": changes,
                            "seeds_with_lower_global_wait": sum(a["network_waiting_s"] < ref["network_waiting_s"] for a, ref in pairs),
                            "seeds_with_lower_secondary_wait": sum(a["secondary_total_s"] < ref["secondary_total_s"] for a, ref in pairs),
                            "seeds_with_no_stop_at_least_reference": sum(a["no_stop_full"] >= ref["no_stop_full"] for a, ref in pairs)}
        lines += ["", "Las comparaciones son emparejadas por demanda; un mejor agregado puede coexistir con pérdidas "
                  "en una demanda individual. Estas mismas cinco demandas guiaron las iteraciones: no son una validación "
                  "independiente en tráfico desconocido."]
        if chosen["means"]["full_waiting_s"] > aggregates["v3"]["means"]["full_waiting_s"]:
            lines += ["", f"Límite restante: el waiting de la cohorte completa sube de "
                      f"{aggregates['v3']['means']['full_waiting_s']:.1f} a {chosen['means']['full_waiting_s']:.1f} "
                      "veh·s por corrida, aunque mejoren la progresión y el tiempo medio. "
                      "La demanda 42 concentra la mayor parte de esa espera; no todas las métricas individuales mejoran."]
    lines += ["", "## Variantes y límites", "",
              "- V3.1 cambia sólo la distribución del adelanto en J2: primero recorta las fases pendientes con menos "
              "vehículos detenidos; desempata por vehículos presentes y orden original. No reordena fases ni modifica el adelanto total. "
              "J10/J16 conservan V3.", "",
              "- V3.2 añade a V3.1 el cierre temprano sólo en J2: liberación terminada, cola objetivo ya cubierta, "
              "ninguna solicitud pendiente distinta y E1 completamente vacío. No supone tamaño de pelotón; "
              "el amarillo se ejecuta por SUMO, sin saltar fases.", "",
              "- V3.3 parte de V3.1, sin cierre temprano: conserva el aviso pero difiere la aplicación de recortes de J2 "
              "hasta la última oportunidad legal con 1 s de margen. El primer cruce real puede corregir el frente antes "
              "de comprometer un recorte; se cierran solicitudes vacías o vencidas.", "",
              "- V3.4 parte de V3.1 y aplica el mismo reparto por colas en J2/J10/J16. Sólo amplía el ámbito "
              "de ese reparto; no incluye cierre temprano ni preparación diferida. Conserva los objetivos 12/22/32 s.", "",
              "- V3.5 añade a V3.4 la corrección de aperturas demasiado tempranas: si el siguiente verde natural "
              "comienza al menos 15 s antes del frente, puede alargar legalmente verdes no-corredor pendientes "
              "(máximo 45 s), dando primero tiempo a las colas mayores. Intenta abrir 1 s antes del target por "
              "el muestreo, sin cambiar el target ni reservar la duración del próximo verde PPO.", "",
              "- V3.6 añade a V3.5 un margen de apertura de 1 s también cuando hay que adelantar el próximo verde. "
              "El target y las ventanas validadas siguen siendo cruce real +12/22/32 s; sólo prepara la apertura "
              "un paso de muestreo antes para cubrir llegadas algo adelantadas.", "",
              "- V3.7 cambia únicamente el margen de V3.6 de 1 a 2 s, para contemplar el muestreo del cruce "
              "de J0 y de la llegada. Los targets causales y el resto del algoritmo permanecen iguales.", "",
              "Se preservan V1/V2/V3, PPO, checkpoint, red y demandas. Validación automática de verdes 5–45 s, "
              "amarillos 4 s, orden cíclico, eventos únicos, frente/cola causales y coherencia de muestreo de 1 s. "
              "Las firmas de las trazas se guardan sin exportar datos crudos. Las tres lane metrics, pendientes, "
              "distribuciones de viaje y configuración se conservan en per_demand.csv."]
    lines += ["", "## Ejecución", "", "```powershell",
              f".venv\\Scripts\\python.exe run_sync.py --mode {recommended or 'v3.1'} --seed 42 --gui",
              "```", "",
              "También se pueden ejecutar v1, v2, v3 y todas las variantes conservadas v3.1–v3.7. "
              "Sin --gui se ejecuta headless; --traces exporta datos detallados sólo cuando se pide explícitamente."]
    (REPORT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(REPORT_DIR / "per_demand.csv", rows)
    archived_path = REPORT_DIR / "comparison.json"
    archived = json.loads(archived_path.read_text(encoding="utf-8")) if archived_path.exists() else {}
    mechanisms = dict(archived.get("mechanism_checks", {}))
    mechanisms.update({p.stem: json.loads(p.read_text(encoding="utf-8"))["validation"]
                       for p in sorted(RUNS_DIR.glob("seed*_v3*.json"))})
    validation = {"aggregates": aggregates, "constraints_zero": all(r["constraint_violations"] == 0 for r in rows),
                  "teleports_zero": all(r["teleports"] == 0 for r in rows),
                  "corridor_pending_zero": all(r["corridor_pending"] == 0 for r in rows),
                  "mechanism_checks": mechanisms}
    reference = json.loads((ROOT / "analysis/corridor_full/reference.json").read_text(encoding="utf-8"))
    protected(reference)
    original_path = RUNS_DIR / "seed42_v3.json"
    if original_path.exists():
        original = json.loads(original_path.read_text(encoding="utf-8"))
        compare(original["summary"], reference["runs"]["seed42_v3_full"]["summary"])
        compare(original["validation"]["canonical_traces"], reference["runs"]["seed42_v3_full"]["traces"])
    validation.update(recommended=recommended, paired_comparison=paired, protected_resources_unchanged=True,
                      source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      configuration={"horizon_s": 3600, "SUMO_seed": 42, "demand_seeds": list(range(42, 47)),
                                     "offsets": [0, 0, 72, 24], "delays": [12, 22, 32],
                                     "checkpoint": "policy/models/model_future_v39_yellow_test36_3.pth",
                                     "green_bounds": [5, 45], "yellow_s": 4, "step_s": 1,
                                     "deterministic": True, "training": False},
                      original_v3_seed42_summary_and_seven_traces_equal=(original_path.exists() or
                          archived.get("original_v3_seed42_summary_and_seven_traces_equal", False)),
                      code_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [ROOT / "run_sync.py", *sorted((ROOT / "sync").glob("*.py"))]})
    original_checks = ROOT / "results/v3_balance_originals/validation.json"
    if original_checks.exists():
        validation["original_v1_v2_replays"] = json.loads(original_checks.read_text(encoding="utf-8"))
    elif "original_v1_v2_replays" in archived:
        validation["original_v1_v2_replays"] = archived["original_v1_v2_replays"]
    native_checks = RUNS_DIR / "native_checks.json"
    if native_checks.exists():
        validation["native_mechanism_validation"] = json.loads(native_checks.read_text(encoding="utf-8"))
    elif "native_mechanism_validation" in archived:
        validation["native_mechanism_validation"] = archived["native_mechanism_validation"]
    (REPORT_DIR / "comparison.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT / "analysis/corridor_full/reference.json").read_text(encoding="utf-8"))
    protected(reference)
    original_path = RUNS_DIR / "seed42_v3.json"
    if args.mode != "v3" and not original_path.exists():
        print("Verify original V3 before the new experiment", flush=True)
        original = run(42, "v3")
        checks = evidence(original)
        compare(original[0], reference["runs"]["seed42_v3_full"]["summary"])
        compare(checks["canonical_traces"], reference["runs"]["seed42_v3_full"]["traces"])
        original_path.write_text(json.dumps({"summary": original[0], "validation": checks}, indent=2), encoding="utf-8")
    for seed in args.seeds:
        path = RUNS_DIR / f"seed{seed}_{args.mode}.json"
        if path.exists():
            raise FileExistsError(path)
        print(f"START {args.mode} demand={seed}", flush=True)
        result = run(seed, args.mode)
        summary = result[0]
        current = evidence(result)
        if args.mode == "v3":
            expected = reference["runs"][f"seed{seed}_v3_full"]
            compare(summary, expected["summary"])
            compare(current["canonical_traces"], expected["traces"])
        else:
            cohort = sorted(r["vehicle_id"] for r in result[-1] if r["exit_J16"] is not None)
            assert cohort == expected_cohort(seed), "Vehicle cohorts changed"
        path.write_text(json.dumps({"summary": summary, "validation": current}, indent=2), encoding="utf-8")
        publish()
        full = summary["full_corridor_any_exit"]
        print(f"DONE {args.mode} demand={seed}: {full['no_stop_all_three']}/{full['cohort']}; "
              f"global={summary['network_total_wait_s']}; secondary={summary['controlled_secondary_wait_s']}", flush=True)
    protected(reference)


if __name__ == "__main__":
    main()
