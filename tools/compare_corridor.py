"""Matched six-configuration experiment; active controllers remain unchanged."""
import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sync.common import CHECKPOINT, DEMAND, END, NET, OFFSETS, ROOT, TLS, static_net, sumo_binary, traci
from sync.metrics import FullObserver, add_audit, distribution, overlap, validate_phases, write_csv
from sync.simulation import (build_agent, run, _normalized_action_to_duration,
                             _structured_snapshot, _yellow_state_from_green_state)
from tests.collect_validation import compact
from tests.validate_sync import canonical, compare, protected
from run_sync import TRACE_NAMES

MODES = ("plain", "plain_offset", "only_ppo", "v1", "v2", "v3")
LABELS = {"plain": "Plain - verde fijo", "plain_offset": "Plain - offsets fijos",
          "only_ppo": "Solo PPO en J0", "v1": "V1 - adelanto por inicio",
          "v2": "V2 - copia de verde PPO", "v3": "V3 - preaviso y ventana causal"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def baseline(seed, mode):
    """Observe fixed programs, or apply only the original deterministic J0 PPO."""
    route = DEMAND / f"seed_{seed}.rou.xml"
    offsets = OFFSETS if mode == "plain_offset" else (0, 0, 0, 0)
    with tempfile.TemporaryDirectory(prefix=f"sumo_compare_{mode}_") as tmp:
        net = Path(tmp) / "wave.net.xml"
        static_net(net, offsets)
        traci.start([sumo_binary(), "--net-file", str(net), "--route-files", str(route),
                     "--begin", "0", "--end", str(END), "--step-length", "1", "--seed", "42",
                     "--no-step-log", "true", "--no-warnings", "true"])
        try:
            programs = {tls: traci.trafficlight.getProgram(tls) for tls in TLS}
            signatures = {tls: [[(p.duration, p.state) for p in logic.phases]
                               for logic in traci.trafficlight.getAllProgramLogics(tls)] for tls in TLS}
            if set(programs.values()) != {"0"} or any(s != signatures["J0"] or len(s) != 1 for s in signatures.values()):
                raise RuntimeError("Incorrect fixed programs")
            if [d for d, _ in signatures["J0"][0]] != [15, 4] * 4:
                raise RuntimeError("Expected four 15 s greens and four 4 s yellows")
            obs = FullObserver(route)
            actions, j0_runs = [], []
            if mode == "only_ppo":
                agent, phases, lanes, indices, _ = build_agent()
                cursor = 0
                while traci.simulation.getTime() < END and traci.simulation.getMinExpectedNumber() > 0:
                    state = _structured_snapshot(lanes, indices, phase_idx=cursor)["state"]
                    action, _, _, _ = agent.act(state, deterministic=True, update_rms=False)
                    decided = _normalized_action_to_duration(action, min_green=5, max_green=45)
                    green = min(decided, int(END - traci.simulation.getTime()))
                    start = int(traci.simulation.getTime()) + 1
                    traci.trafficlight.setRedYellowGreenState("J0", phases[cursor]["state"])
                    actions.append({"phase_cursor": cursor, "phase_xml": phases[cursor]["index"],
                                    "decision_at": start - 1, "start": start, "green": green,
                                    "decided_green": decided, "action": float(np.asarray(action).item())})
                    if cursor == 1:
                        obs.begin_release(start, green)
                    for second in range(green):
                        traci.simulationStep()
                        obs.sample()
                        if cursor == 1 and second == 0:
                            obs.J0_releases.append(start)
                    j0_runs.append({"tls": "J0", "phase": phases[cursor]["index"], "start": start,
                                    "duration": green, "requested": decided, "partial": green != decided})
                    if cursor == 1:
                        obs.end_release()
                    remaining = int(END - traci.simulation.getTime())
                    if remaining <= 0 or traci.simulation.getMinExpectedNumber() <= 0:
                        break
                    traci.trafficlight.setRedYellowGreenState("J0", _yellow_state_from_green_state(phases[cursor]["state"]))
                    yellow_start = int(traci.simulation.getTime()) + 1
                    yellow = min(4, remaining)
                    for _ in range(yellow):
                        traci.simulationStep()
                        obs.sample()
                    j0_runs.append({"tls": "J0", "phase": phases[cursor]["index"] + 1,
                                    "start": yellow_start, "duration": yellow, "partial": yellow != 4})
                    cursor = (cursor + 1) % 4
            else:
                phase, started = traci.trafficlight.getPhase("J0"), 1
                while traci.simulation.getTime() < END and traci.simulation.getMinExpectedNumber() > 0:
                    traci.simulationStep()
                    now = int(traci.simulation.getTime())
                    current = traci.trafficlight.getPhase("J0")
                    if current != phase:
                        j0_runs.append({"tls": "J0", "phase": phase, "start": started,
                                        "ended_at": now, "duration": now - started, "partial": False})
                        if phase == 2:
                            obs.end_release()
                        phase, started = current, now
                        if current == 2:
                            obs.begin_release(now, 15)
                            obs.J0_releases.append(now)
                    obs.sample()
                now = int(traci.simulation.getTime()) + 1
                j0_runs.append({"tls": "J0", "phase": phase, "start": started,
                                "ended_at": now, "duration": now - started, "partial": True})
                obs.end_release()
            obs.finish()
            validate_phases({"J0": j0_runs, **obs.runs})
            for tls, rows in obs.runs.items():
                if any(row["duration"] != (4 if row["phase"] % 2 else 15)
                       for row in rows if not row["partial"]):
                    raise RuntimeError(f"{tls}: fixed receiver was changed")
            strict = {v: r for v, r in obs.records.items() if v in obs.cohort and r["E5_start"] is not None}
            summary = {**obs.lane_metrics(), "seed": seed, "mode": mode, "coordination_scope": "full",
                       "horizon_s": END, "SUMO_seed": 42, "cumulative_delays_s": {},
                       "offsets": list(offsets), "runtime_programs": programs,
                       "network_sha256": hashlib.sha256(NET.read_bytes()).hexdigest(),
                       "route_sha256": hashlib.sha256(route.read_bytes()).hexdigest(),
                       "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() if mode == "only_ppo" else None,
                       "cohort": len(obs.cohort), "crossed": len(strict),
                       "no_stop": sum(not r["stopped_E1"] for r in strict.values()),
                       "no_stop_pct": 100 * sum(not r["stopped_E1"] for r in strict.values()) / len(obs.cohort),
                       "waiting_E1_s": sum(r["E1_wait"] for r in strict.values()),
                       "mean_E1_to_E5_s": statistics.mean(r["E5_start"] - r["E1_start"] for r in strict.values()),
                       "secondary_wait_J2_s": obs.secondary_wait, "network_total_wait_s": obs.network_wait,
                       "tls_wait_s": obs.tls_wait, "arrived": len(obs.arrived), "departed": len(obs.departed),
                       "teleported": len(obs.teleports), "pending": {
                           "min_expected": traci.simulation.getMinExpectedNumber(),
                           "active": len(traci.vehicle.getIDList()),
                           "pending_insertion": len(traci.simulation.getPendingVehicles())}, "receivers": {}}
            per_vehicle = add_audit(summary, obs)
            return (summary, obs.release_windows, list(obs.audit.vehicles.values()), [],
                    obs.full_timeline, actions, j0_runs + [r for rows in obs.runs.values() for r in rows], per_vehicle)
        finally:
            traci.close()


def physical_coverage(result):
    summary, _, source_rows, _, _, _, phases, vehicles = result
    sources = {r["vehicle_id"]: r for r in source_rows}
    groups, cohort_sets = {}, {}
    for row in vehicles:
        release = sources.get(row["vehicle_id"], {}).get("release_id")
        if release is None:
            continue
        for tls in ("J2", "J10", "J16"):
            if row.get(f"{tls}_arrival_at") is not None:
                groups.setdefault((tls, release), []).append(row)
    for tls in ("J2", "J10", "J16"):
        greens = []
        for p in phases:
            if p["tls"] == tls and p["phase"] == (4 if tls == "J10" else 2):
                start = p.get("start", p.get("ended_at", 0) - p["duration"])
                end = p.get("ended_at", start + p["duration"])
                greens.append((start, end))
        spans = [(min(r[f"{tls}_arrival_at"] for r in members), max(r[f"{tls}_arrival_at"] for r in members) + 1)
                 for (target, _), members in groups.items() if target == tls]
        total = sum(b - a for a, b in spans)
        cohort_sets[tls] = {"windows": len(spans), "arrival_window_s": total,
                            "green_covered_s": sum(overlap(a, b, greens) for a, b in spans),
                            "coverage_pct": 100 * sum(overlap(a, b, greens) for a, b in spans) / total if total else None}
    return cohort_sets


def aggregate(rows):
    answer = {}
    for mode in MODES:
        items = [r for r in rows if r["mode"] == mode]
        total = {k: sum(r[k] for r in items) for k in ("no_stop_full", "cohort_full", "no_stop_J2", "cohort_J2", "full_waiting_s", "waiting_E1_s", "strict_no_stop", "strict_cohort", "corridor_pending")}
        metrics = {}
        for key in ("network_waiting_s", "secondary_total_s", "secondary_J2_s", "secondary_J10_s", "secondary_J16_s",
                    "mean_J0_to_J16_s", "stops_per_vehicle", "mean_E1_to_E5_s", "pending", "arrived", "waiting_time", "effective_flow", "avg_queue_length"):
            values = [r[key] for r in items]
            metrics[key] = {"mean": statistics.mean(values), "sd": statistics.stdev(values), "min": min(values), "max": max(values)}
        answer[mode] = {"label": LABELS[mode], "totals": total, "across_demands": metrics,
                        "full_no_stop_pct": 100 * total["no_stop_full"] / total["cohort_full"],
                        "J2_no_stop_pct": 100 * total["no_stop_J2"] / total["cohort_J2"]}
    return answer


def report(output, rows, aggregates):
    lines = ["# Comparación de seis configuraciones", "",
             "Demandas 42–46; horizonte 3600 s; paso 1 s; SUMO seed 42. Ejecución secuencial en el orden solicitado.", "",
             "Plain usa cuatro verdes de 15 s y amarillos de 4 s, offsets cero. Plain con offset usa 0/0/72/24. "
             "Sólo PPO usa el checkpoint oficial determinista en J0 y receptores fijos con offsets cero. "
             "V1/V2/V3 conservan su implementación y sus offsets base 0/0/72/24.", "",
             "| Configuración | Sin parada J2/J10/J16 | Paradas/veh. | Media J0→J16 (s) | Waiting E1 total | Waiting corredor total | Waiting secundario medio | Waiting red medio ± DE | Pendientes red medios |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        a = aggregates[mode]
        t, m = a["totals"], a["across_demands"]
        lines.append(f"| {a['label']} | {t['no_stop_full']}/{t['cohort_full']} ({a['full_no_stop_pct']:.2f}%) | "
                     f"{m['stops_per_vehicle']['mean']:.3f} | {m['mean_J0_to_J16_s']['mean']:.2f} | {t['waiting_E1_s']} | "
                     f"{t['full_waiting_s']} | {m['secondary_total_s']['mean']:.1f} | {m['network_waiting_s']['mean']:.1f} ± "
                     f"{m['network_waiting_s']['sd']:.1f} | {m['pending']['mean']:.1f} |")
    lines += ["", "## Resultados por demanda", "",
              "| Configuración | Demanda | Sin parada completa | Media J0→J16 | Waiting E1 | Waiting secundario | Waiting red | Pendientes red/corredor |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['seed']} | {r['no_stop_full']}/{r['cohort_full']} | {r['mean_J0_to_J16_s']:.2f} | "
                     f"{r['waiting_E1_s']} | {r['secondary_total_s']} | {r['network_waiting_s']} | {r['pending']}/{r['corridor_pending']} |")
    lines += ["", "## Definiciones y comprobaciones", "",
              "Cohorte completa: -E0→E1→E5→E10 y cruce de J16, incluida cualquier salida posterior; depart ≤3420 s. "
              "Salida recta E13 se registra aparte. Sin parada: velocidad <0,1 m/s nunca observada en E1/E5/E10. "
              "Paradas por vehículo cuentan episodios en esos enlaces; J0→J16 empieza al cruce real de J0, excluyendo la espera previa.", "",
              "Waiting en veh·s. E1 total corresponde a la cohorte recta J0→J2; waiting corredor a la cohorte completa sobre E1/E5/E10. "
              "Waiting secundario suma aproximaciones no-corredor de J2/J10/J16, incluido el sentido contrario; excluye J0. "
              "Waiting red cuenta todos los vehículos detenidos, incluidas conexiones internas. "
              "Tiempos y paradas del resumen son medias de las medias por demanda; porcentajes son ponderados por vehículos. "
              "Waiting red y secundario son medias de cinco corridas; E1 y corredor en esta tabla son sumas de las cinco.", "",
              "Las tres lane metrics se conservan en per_demand.csv: waiting_time (carriles controlados únicos), "
              "effective_flow (veh·s en movimiento, no throughput) y avg_queue_length (vehículos/carril). "
              "La cobertura física de ventanas se calcula con llegadas observadas, incluyendo cotas superiores a resolución de 1 s; no equivale a progresión sin parada.", "",
              "V1/V2/V3 se comparan con reference.json antes de añadir etiquetas: métricas y siete firmas canónicas exactas. "
              "Programas iniciales 0 de 76 s, orden cíclico, amarillos completos de 4 s y verdes 5–45 s comprobados; "
              "fragmentos iniciales/finales censurados. Las bases fijas verifican además verdes de exactamente 15 s. "
              "Se usan los mismos vehículos de cohorte en las seis configuraciones; código, red, checkpoint y demandas protegidos por hashes."]
    lines += ["", "## Primer tramo y métricas de carriles", "",
              "Medias por corrida; las cohortes J0→J2 suman 232 vehículos, mientras la cohorte que recorre todo el corredor suma 93.", "",
              "| Configuración | Sin parada J2 | Tiempo E1→E5 (s) | Waiting E1 medio | Waiting corredor medio | Lane waiting | Effective flow | Cola media/carril |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        a = aggregates[mode]
        t, m = a["totals"], a["across_demands"]
        lines.append(f"| {a['label']} | {t['no_stop_J2']}/{t['cohort_J2']} ({a['J2_no_stop_pct']:.2f}%) | "
                     f"{m['mean_E1_to_E5_s']['mean']:.2f} | {t['waiting_E1_s']/5:.1f} | {t['full_waiting_s']/5:.1f} | "
                     f"{m['waiting_time']['mean']:.1f} | {m['effective_flow']['mean']:.1f} | {m['avg_queue_length']['mean']:.4f} |")
    lines += ["", "## Interpretación", "",
              "- La onda fija con offsets obtiene 79/93 recorridos sin parada frente a 0/93 sin offsets. "
              "Reduce el tiempo medio de 85,47 a 40,40 s y el waiting global un 3,39%, pero aumenta el waiting secundario un 14,51%.",
              "- Sólo PPO con receptores sin offsets mantiene 0/93 recorridos sin parada. El tiempo medio aumenta a 114,99 s "
              "y la espera del corredor a 1281,2 veh·s por corrida; el waiting global se mantiene próximo al plain.",
              "- V1 tiene el menor waiting global (44071,6 veh·s): 12,16% menos que plain y 11,88% menos que sólo PPO. "
              "Su progresión completa queda en 54,84%, por debajo de la onda fija con offsets.",
              "- V2 eleva la progresión frente a V1 a 65,59% y consigue el menor waiting secundario (25328 veh·s), "
              "pero su tiempo medio (42,54 s) y waiting del corredor (127,6 veh·s por corrida) superan los de V1. "
              "El porcentaje sin parada y el promedio de espera describen aspectos distintos de la distribución.",
              "- V3 logra la mayor progresión (84/93, 90,32%) y el menor tiempo medio (36,09 s). "
              "Frente a la onda fija añade 5 vehículos sin parada y reduce el tiempo un 10,65%, "
              "pero aumenta el waiting global un 2,20% y el secundario un 10,37%. "
              "Frente a V1, su waiting global aumenta un 12,45%.",
              "- Ningún vehículo de la cohorte completa queda pendiente; sí quedan vehículos de la red al terminar los 3600 s. "
              "V3 presenta el mayor promedio de pendientes (30,4), V2 el menor (24,8).", "",
              "Las cifras describen estas cinco demandas. La comparación sólo PPO frente a V1/V2/V3 cambia tanto la coordinación "
              "como los offsets base, conforme a las configuraciones solicitadas; no aísla el efecto del controlador por sí solo. "
              "La comparación plain frente a plain con offsets sí mantiene los demás factores constantes.", "",
              "V1: aviso desde inicio del verde, objetivos acumulados 11/21/31 s y recorte de verdes no-corredor. "
              "V2: copia de la duración PPO, objetivos 11/21/31 s. V3: preaviso desde la decisión del verde anterior, "
              "corrección del frente y cola según cruces reales, desplazamientos 12/22/32 s. "
              "Los tres coordinan J2/J10/J16; PPO controla únicamente J0 con policy/models/model_future_v39_yellow_test36_3.pth.", "",
              "Recomendación: mantener plain con offsets como referencia de onda fija, V1 como referencia de waiting global, "
              "V2 como referencia de coste secundario y V3 como referencia de progresión. Estas corridas no muestran una versión "
              "que gane simultáneamente en todos los objetivos."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/comparison_six")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "analysis/six_configurations")
    parser.add_argument("--traces", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference = json.loads((ROOT / "analysis/corridor_full/reference.json").read_text(encoding="utf-8"))
    protected(reference)
    code_paths = [ROOT / "run_sync.py", *sorted((ROOT / "sync").glob("*.py")), *sorted((ROOT / "tests").glob("*.py"))]
    code_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    records, order, cohort_by_seed, full_records = [], [], {}, []
    for mode in MODES:
        for seed in range(42, 47):
            started = utc()
            print(f"START {mode} seed={seed} {started}", flush=True)
            result = run(seed, mode) if mode in ("v1", "v2", "v3") else baseline(seed, mode)
            summary = result[0]
            signatures = {name: canonical(values) for name, values in zip(TRACE_NAMES, result[1:])}
            matched = None
            if mode in ("v1", "v2", "v3"):
                expected = reference["runs"][f"seed{seed}_{mode}_full"]
                compare(summary, expected["summary"])
                compare(signatures, expected["traces"])
                matched = True
            cohort = sorted(r["vehicle_id"] for r in result[-1] if r["exit_J16"] is not None)
            if seed not in cohort_by_seed:
                cohort_by_seed[seed] = cohort
            if cohort != cohort_by_seed[seed]:
                raise RuntimeError("Comparison cohorts differ")
            covered = physical_coverage(result)
            row = compact(summary)
            row.update(order=MODES.index(mode) + 1, label=LABELS[mode], reference_equal=matched,
                       checkpoint=row["checkpoint"] if mode not in ("plain", "plain_offset") else None,
                       green_min_s=15 if mode.startswith("plain") else 5,
                       green_max_s=15 if mode.startswith("plain") else 45)
            for tls, metrics in covered.items():
                row[f"physical_window_coverage_{tls}_pct"] = metrics["coverage_pct"]
            records.append(row)
            saved = {"summary": summary, "physical_windows": covered, "canonical_traces": signatures,
                     "reference_equal": matched, "cohort_ids": cohort, "label": LABELS[mode]}
            full_records.append(saved)
            (args.output_dir / f"seed{seed}_{mode}.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
            if args.traces:
                for name, values in zip(TRACE_NAMES, result[1:]):
                    write_csv(args.output_dir / f"seed{seed}_{mode}_{name}.csv", values)
            order.append({"mode": mode, "seed": seed, "started_UTC": started, "finished_UTC": utc()})
            write_csv(args.output_dir / "per_demand.csv", records)
            (args.output_dir / "execution_order.json").write_text(json.dumps(order, indent=2), encoding="utf-8")
            print(f"DONE {mode} seed={seed}: full={row['no_stop_full']}/{row['cohort_full']}; "
                  f"network_wait={row['network_waiting_s']}; pending={row['pending']}; reference={matched}", flush=True)
    protected(reference)
    assert code_hashes == {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    args.report_dir.mkdir(parents=True, exist_ok=False)
    aggregates = aggregate(records)
    for mode in MODES:
        for tls in ("J2", "J10", "J16"):
            windows = [r["physical_windows"][tls] for r in full_records if r["summary"]["mode"] == mode]
            aggregates[mode].setdefault("physical_window_coverage", {})[tls] = 100 * sum(w["green_covered_s"] for w in windows) / sum(w["arrival_window_s"] for w in windows)
    write_csv(args.report_dir / "per_demand.csv", records)
    (args.report_dir / "aggregate.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    validation = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "execution_order": order, "matched_vehicle_sets": True, "protected_resources_unchanged": True,
                  "protected_code_sha256": code_hashes, "sync_runs_reference_equal": 15,
                  "runs": [{"mode": r["mode"], "seed": r["seed"], "constraint_violations": r["constraint_violations"],
                            "teleports": r["teleports"], "corridor_pending": r["corridor_pending"]} for r in records]}
    (args.report_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    report(args.report_dir, records, aggregates)
    print(f"FINISHED: {args.report_dir}", flush=True)


if __name__ == "__main__":
    main()
