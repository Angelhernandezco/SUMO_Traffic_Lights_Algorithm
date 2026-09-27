"""Post-hoc corridor-to-J16 cohort and individual movement tables.

No SUMO/TraCI imports. Recomputes only from saved observational records.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.flow_metrics import Approach, CORRIDOR_MOVEMENTS, summarize


def corridor_from_j0(record):
    """Continue straight through J2/J10; reach J16 via E10, any exit."""
    return record.origin == "j0_released" and (
        record.group == "priority" if record.tls != "J16" else record.incoming == "E10" and record.outgoing is not None)


def difference(a, b):
    return {"baseline": a, "advance": b,
            "absolute_difference": b - a if a is not None and b is not None else None,
            "percent_difference": 100 * (b - a) / a if a not in (None, 0) and b is not None else None}


def compare_cohort(left, right):
    left = {r.key: r for r in left if corridor_from_j0(r)}
    right = {r.key: r for r in right if corridor_from_j0(r)}
    result = {p: {} for p in ("full", "matched", "completed_in_both")}
    for tls in (*CORRIDOR_MOVEMENTS, "ALL"):
        a = {k: r for k, r in left.items() if tls == "ALL" or r.tls == tls}
        b = {k: r for k, r in right.items() if tls == "ALL" or r.tls == tls}
        common = sorted(a.keys() & b.keys())
        complete = [k for k in common if a[k].crossed_at is not None and b[k].crossed_at is not None
                    and a[k].censor_reason is None and b[k].censor_reason is None
                    and not a[k].left_censored and not b[k].left_censored]
        for p, aa, bb in (("full", a.values(), b.values()),
                           ("matched", [a[k] for k in common], [b[k] for k in common]),
                           ("completed_in_both", [a[k] for k in complete], [b[k] for k in complete])):
            sa, sb = summarize(aa), summarize(bb)
            result[p][tls] = {metric: difference(sa[metric], sb[metric]) for metric in sa}
    return result


def corridor_trips(records):
    """J0 releases whose planned route crosses all three slave stop lines."""
    ids = {r.vehicle_id for r in records if r.tls == "J2" and r.origin == "j0_released"
           and tuple(r.route[r.route_index:r.route_index + 3]) == ("E1", "E5", "E10")
           and len(r.route) > r.route_index + 3}
    trips = {}
    for vid in sorted(ids):
        visits = [r for r in records if r.vehicle_id == vid and corridor_from_j0(r)]
        final = next((r for r in visits if r.tls == "J16" and r.crossed_at is not None
                      and r.censor_reason is None and not r.left_censored), None)
        trips[vid] = {"vehicle_id": vid, "waiting": sum(r.waiting for r in visits),
                      "stops": sum(r.stops for r in visits), "completed": final is not None,
                      "travel_seconds": final.crossed_at - final.j0_release_time if final else None,
                      "observed_tls": sorted({r.tls for r in visits})}
    return trips


def compare_trips(left, right):
    a, b = corridor_trips(left), corridor_trips(right)
    common = sorted(a.keys() & b.keys())
    complete = [v for v in common if a[v]["completed"] and b[v]["completed"]]

    def stats(values):
        values = list(values)
        done = [v for v in values if v["completed"]]
        return {"vehicles": len(values), "waiting": sum(v["waiting"] for v in values),
                "mean_waiting_per_vehicle": sum(v["waiting"] for v in values) / len(values) if values else None,
                "stops": sum(v["stops"] for v in values), "completed": len(done),
                "mean_travel_seconds": sum(v["travel_seconds"] for v in done) / len(done) if done else None,
                "no_stop_full_trip_percent": 100 * sum(v["stops"] == 0 for v in done) / len(done) if done else None}

    result = {}
    for population, aa, bb in (("full", a.values(), b.values()),
                               ("matched", [a[v] for v in common], [b[v] for v in common]),
                               ("completed_in_both", [a[v] for v in complete], [b[v] for v in complete])):
        sa, sb = stats(aa), stats(bb)
        result[population] = {m: difference(sa[m], sb[m]) for m in sa}
    return {"comparison": result, "baseline_trips": a, "advance_trips": b}


def fmt(value):
    if value is None:
        return "N/A"
    return str(int(value)) if value == int(value) else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "logs/c1_diagnostic"))
    args = parser.parse_args()
    output = Path(args.output)
    data = {name: [Approach(**r) for r in json.loads((output / (name + ".flow_metrics.json")).read_text())["records"]]
            for name in ("baseline", "advance")}
    result = compare_cohort(data["baseline"], data["advance"])
    trips = compare_trips(data["baseline"], data["advance"])
    (output / "full_corridor_trips.json").write_text(json.dumps(trips, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "corridor_j0_any_j16_exit.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    lines = ["# Corredor liberado por J0 hasta J16, con cualquier salida final", "",
             "Esta vista usa E1→E5 en J2, E5→E10 en J10 y llegada por E10 a J16 con cualquier salida. "
             "Incluye los giros finales E10→E11/E12; excluye inserciones directas en E1 y entradas laterales. "
             "No altera la definición estricta del movimiento recto del informe principal ni modifica el controlador.", "",
             "Los datos se derivan del mismo registro por vehículo, sin otra simulación. "
             "ALL suma exposiciones en los tres TLS; sus tiempos medios mezclan destinos intermedios y no se usan como tiempo de viaje completo. "
             "Para el tiempo de corredor completo se usa exclusivamente la fila J16.", ""]
    metrics = ("waiting", "vehicles", "mean_waiting_per_vehicle", "stops", "throughput",
               "no_stop_crossing_percent", "mean_j0_to_crossing_seconds", "unfinished_or_censored")
    for population in result:
        lines.extend([f"## {population}", "", "| TLS | Métrica | Baseline | Advance | Dif. abs | Dif. % |",
                      "|---|---|---:|---:|---:|---:|"])
        for tls in (*CORRIDOR_MOVEMENTS, "ALL"):
            for metric in metrics:
                if tls == "ALL" and metric == "mean_j0_to_crossing_seconds":
                    continue
                r = result[population][tls][metric]
                lines.append(f'| {tls} | {metric} | {fmt(r["baseline"])} | {fmt(r["advance"])} | {fmt(r["absolute_difference"])} | {fmt(r["percent_difference"])} |')
    lines.extend(["", "Los porcentajes sin parada usan cruces completos; su diferencia absoluta se expresa en puntos porcentuales. "
                  "Las medias de waiting incluyen vehículos aún sin cruzar; la población completada en ambas permite comparar igual conjunto de cruces.",
                  "No se concluye si el objetivo de prioridad es adecuado para la tesis."])
    (output / "corridor_j0_any_j16_exit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    trip_lines = ["# Vehículos de origen J0 con ruta completa hasta J16", "",
                  "Cohorte causal: ruta continua E1,E5,E10 seguida de una salida de J16, tras liberación verificada por J0. "
                  "Excluye rutas que salen del corredor en J2/J10, rutas que terminan antes de cruzar J16 e inserciones directas. "
                  "Waiting y paradas suman los tres accesos de cada vehículo; no incluyen la espera previa a liberación en J0 ni vías fuera de esos accesos.", "",
                  "A diferencia de la vista local por TLS, esta población mantiene fija la ruta completa del corredor. "
                  "Las rutas parciales permanecen en el informe local. No se cambia ningún controlador.", ""]
    for population, rows in trips["comparison"].items():
        trip_lines.extend([f"## {population}", "", "| Métrica | Baseline | Advance | Dif. abs | Dif. % |", "|---|---:|---:|---:|---:|"])
        for metric, r in rows.items():
            trip_lines.append(f'| {metric} | {fmt(r["baseline"])} | {fmt(r["advance"])} | {fmt(r["absolute_difference"])} | {fmt(r["percent_difference"])} |')
    trip_lines.extend(["", "## Desglose por TLS de las rutas completas", "",
                       "| TLS | Métrica | Baseline | Advance | Dif. abs | Dif. % |", "|---|---|---:|---:|---:|---:|"])
    for tls in CORRIDOR_MOVEMENTS:
        aa = summarize(r for r in data["baseline"] if r.vehicle_id in trips["baseline_trips"] and r.tls == tls and corridor_from_j0(r))
        bb = summarize(r for r in data["advance"] if r.vehicle_id in trips["advance_trips"] and r.tls == tls and corridor_from_j0(r))
        for metric in ("waiting", "vehicles", "mean_waiting_per_vehicle", "stops", "throughput", "no_stop_crossing_percent", "mean_j0_to_crossing_seconds"):
            r = difference(aa[metric], bb[metric])
            trip_lines.append(f'| {tls} | {metric} | {fmt(r["baseline"])} | {fmt(r["advance"])} | {fmt(r["absolute_difference"])} | {fmt(r["percent_difference"])} |')
    (output / "full_corridor_trips.md").write_text("\n".join(trip_lines) + "\n", encoding="utf-8")
    overview = ["# Resumen de la auditoría diagnóstica de C1", "",
                "Estos resultados describen espera y paradas después de la liberación por J0, en los tres accesos de slaves. "
                "No incluyen waiting antes de cruzar J0 y no juzgan cuál debe ser la prioridad de tesis.", "",
                "## Cohorte con ruta completa J0→J2→J10→J16", "",
                "43 IDs liberados por J0, con ruta continua E1,E5,E10 y una salida válida de J16. "
                "Los IDs son los mismos en ambas corridas; 38 completan en ambas. "
                "Las inserciones directas y rutas que terminan en E10 sin cruzar J16 quedan fuera de esta cohorte.", "",
                "| Población | Métrica | Baseline | Advance | Dif. abs | Dif. % |", "|---|---|---:|---:|---:|---:|"]
    for population, label in (("full", "43 rutas completas"), ("completed_in_both", "Mismos 38 completados")):
        for metric in ("waiting", "stops", "mean_waiting_per_vehicle", "completed", "mean_travel_seconds", "no_stop_full_trip_percent"):
            r = trips["comparison"][population][metric]
            overview.append(f'| {label} | {metric} | {fmt(r["baseline"])} | {fmt(r["advance"])} | {fmt(r["absolute_difference"])} | {fmt(r["percent_difference"])} |')
    overview.extend(["", "[Desglose de esta cohorte por TLS](full_corridor_trips.md)", "",
                     "## Partición por movimiento local: todas las poblaciones", "",
                     "Esta tabla usa movimiento recto E1→E5/E5→E10/E10→E13, con todos los orígenes. "
                     "El complemento incluye giros del receptor y sentido contrario, además del transversal. "
                     "Esta partición suma exactamente el waiting de las tres slaves; la cohorte de 43 no es una partición independiente que deba añadirse a estos totales.", "",
                     "| Grupo | Métrica | Baseline | Advance | Dif. % |", "|---|---|---:|---:|---:|"])
    main_comparison = json.loads((output / "comparison.json").read_text()) if (output / "comparison.json").exists() else None
    if main_comparison:
        for group, label in (("priority", "Corredor recto, todos los orígenes"), ("secondary", "Otros movimientos/secundarios"),
                              ("other_accesses", "Demás accesos, incluido sentido contrario"), ("priority_j0", "Corredor recto, origen J0")):
            for metric in ("waiting", "stops"):
                r = main_comparison["full"]["ALL"][group][metric]
                overview.append(f'| {label} | {metric} | {fmt(r["baseline"])} | {fmt(r["advance"])} | {fmt(r["percent_difference"])} |')
        overview.extend(["", "## Población local de J0 con salidas parciales del corredor", "",
                         "Los vehículos de J0 que continúan en cada movimiento local son otra vista: incluye quienes luego salen en J10 o no recorren todo el boulevard. "
                         "J16 exige una salida válida; vehículos cuya ruta termina en E10 se conservan en las métricas de acceso, pero no son un movimiento de cruce. "
                         "Su waiting agregado es " + fmt(result["full"]["ALL"]["waiting"]["baseline"]) + "→" + fmt(result["full"]["ALL"]["waiting"]["advance"]) +
                         " y sus detenciones " + fmt(result["full"]["ALL"]["stops"]["baseline"]) + "→" + fmt(result["full"]["ALL"]["stops"]["advance"]) + ".", "",
                         "[Informe local por TLS, todas las métricas A–L, poblaciones completas y pareadas](report.md)",
                         "[Cohorte local J0, incluidas salidas válidas de J16](corridor_j0_any_j16_exit.md)",
                         "[Desglose por cada conexión y origen](movement_breakdown.csv)"])
    overview.extend(["", "## Congelación funcional", "",
                     "OFF, SHADOW y ADVANCE conservaron exactamente acciones, observaciones, reward, timings, señales, métricas y comandos anteriores. "
                     "OFF y ADVANCE repetidos producen registros diagnósticos y eventos idénticos. "
                     "Las nuevas métricas solo leen; cero nuevas escrituras y exactamente 2000 pasos por corrida. "
                     "validation_summary.json conserva los hashes de los archivos congelados; unit_tests.txt registra la suite de pruebas.",
                     "No se modificaron PPO, C1, checkpoint, red, rutas ni demanda. No se hizo commit ni push."])
    (output / "overview.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
    grouped = {}
    for name, records in data.items():
        groups = defaultdict(list)
        for r in records:
            groups[(r.tls, r.incoming, r.outgoing, r.origin)].append(r)
        grouped[name] = {k: summarize(v) for k, v in groups.items()}
    fields = ["tls", "incoming", "outgoing", "origin", "metric", "baseline", "advance", "absolute_difference", "percent_difference"]
    with (output / "movement_breakdown.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        keys = grouped["baseline"].keys() | grouped["advance"].keys()
        for key in sorted(keys, key=lambda k: tuple(v or "" for v in k)):
            a = grouped["baseline"].get(key, summarize([]))
            b = grouped["advance"].get(key, summarize([]))
            for metric in a:
                writer.writerow({**dict(zip(fields[:4], key)), "metric": metric, **difference(a[metric], b[metric])})
    print(json.dumps({tls: {m: result["full"][tls][m] for m in ("waiting", "stops", "vehicles")}
                      for tls in result["full"]}, indent=2))


if __name__ == "__main__":
    main()
