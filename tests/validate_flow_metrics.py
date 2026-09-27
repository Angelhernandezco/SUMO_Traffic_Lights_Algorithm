"""Frozen C1 diagnostic audit: additional getters only, no controller edits."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import traci
from policy import train
from policy.flow_metrics import Approach, CORRIDOR_MOVEMENTS, FlowMetricsObserver, summarize
from validate_c0 import digest, write_json, sumo_version
from validate_c1 import run as run_c1, audit


class ReadOnlyDomain:
    def __init__(self, domain):
        self.domain = domain

    def __getattr__(self, name):
        if not name.startswith("get"):
            raise AssertionError(f"Telemetry cannot invoke {name}")
        return getattr(self.domain, name)


class ReadOnlyTraCI:
    def __init__(self, api):
        self.vehicle = ReadOnlyDomain(api.vehicle)
        self.lane = ReadOnlyDomain(api.lane)
        self.simulation = ReadOnlyDomain(api.simulation)

    def __getattr__(self, name):
        raise AssertionError(f"Telemetry cannot invoke {name}")


def run(name, mode, output):
    path = output / (name + ".flow_metrics.json")
    complete = output / (name + ".flow_complete.json")
    signature = {"mode": mode, "telemetry": digest(ROOT / "policy/flow_metrics.py"),
                 "validator": digest(Path(__file__)),
                 "frozen": frozen_hashes()}
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["signature"] == signature and all(digest(output / p) == h for p, h in saved["outputs"].items()):
            print(name, "reusing verified diagnostic evidence", flush=True)
            return json.loads((output / (name + ".evaluation.json")).read_text()), json.loads(path.read_text())
    # The original validator does not know about this optional observer.
    # Invalidate only its completion marker if diagnostic evidence is absent.
    marker = output / (name + ".complete.json")
    if marker.exists():
        marker.unlink()
    captured = []
    original = train.SumoTrafficEnv._record_step_metrics
    view = ReadOnlyTraCI(traci)

    def record_step(env):
        result = original(env)
        if not captured:
            captured.append(FlowMetricsObserver(env.contexts))
        captured[0].observe(view, env.forecast, env.c0.local, env.c0.executor)
        return result

    with mock.patch.object(train.SumoTrafficEnv, "_record_step_metrics", autospec=True, side_effect=record_step):
        result = run_c1(name, {"eta_mode": "local", "coordination_mode": mode}, output)
    observer = captured[0]
    diagnostic = {**observer.export(), "summary": observer.summary()}
    write_json(path, diagnostic)
    event_path = output / (name + ".flow_events.jsonl")
    with event_path.open("w", encoding="utf-8", newline="\n") as stream:
        for event in observer.events:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    assert diagnostic["steps"] == 2000 and diagnostic["last_time"] == 2000
    for tls in CORRIDOR_MOVEMENTS:
        row = diagnostic["summary"][tls]
        assert row["all"]["waiting"] == result["metrics"][f"waiting_{tls}"]
        assert row["priority"]["waiting"] + row["secondary"]["waiting"] == row["all"]["waiting"]
        assert row["other_accesses"]["waiting"] == result["supplementary"]["waiting_transverse_slaves"][tls]
    outputs = [name + ".flow_metrics.json", name + ".flow_events.jsonl",
               name + ".evaluation.json", name + ".coordination.jsonl", name + ".baseline_forecast.jsonl"]
    write_json(complete, {"signature": signature, "outputs": {p: digest(output / p) for p in outputs}})
    return result, diagnostic


def frozen_hashes():
    files = ["main.py", "policy/train.py", "policy/coordination.py", "policy/agent.py", "policy/forecast.py",
             "sumo_utils.py", "configuration.sumocfg", "configuration.v39.sumocfg",
             "maps/master_slave.net.xml", "maps/master_slave.rou.xml",
             "policy/models/model_future_v39_yellow_test36_3.pth", "tests/validate_c1.py"]
    return {p: digest(ROOT / p) for p in files}


def load_records(diagnostic):
    return {f'{r["tls"]}:{r["vehicle_id"]}:{r["route_index"]}': Approach(**r) for r in diagnostic["records"]}


def selection(record, group, tls=None):
    if tls is not None and record.tls != tls:
        return False
    receiver = CORRIDOR_MOVEMENTS[record.tls][0]
    if group == "priority_j0":
        return record.group == "priority" and record.origin == "j0_released"
    if group == "priority_direct_e1":
        return record.group == "priority" and record.origin == "direct_e1"
    if group == "priority_other_upstream":
        return record.group == "priority" and record.origin == "other_upstream"
    if group == "receiver_access_all":
        return record.incoming == receiver
    if group == "receiver_j0_all_movements":
        return record.incoming == receiver and record.origin == "j0_released"
    if group == "receiver_turns_or_terminal":
        return record.incoming == receiver and record.group != "priority"
    if group == "other_accesses":
        return record.incoming != receiver
    return record.group == group


GROUPS = ("priority", "secondary", "priority_j0", "priority_direct_e1", "priority_other_upstream",
          "receiver_access_all", "receiver_j0_all_movements", "receiver_turns_or_terminal", "other_accesses")


def compare(baseline, advance):
    a, b = load_records(baseline), load_records(advance)
    tables = {"full": {}, "matched": {}, "completed_in_both": {}}
    for tls in (*CORRIDOR_MOVEMENTS, "ALL"):
        target = None if tls == "ALL" else tls
        for table in tables.values():
            table[tls] = {}
        for group in GROUPS:
            aa = {k: r for k, r in a.items() if selection(r, group, target)}
            bb = {k: r for k, r in b.items() if selection(r, group, target)}
            common = sorted(aa.keys() & bb.keys())
            completed = [k for k in common if aa[k].crossed_at is not None and bb[k].crossed_at is not None
                         and aa[k].censor_reason is None and bb[k].censor_reason is None
                         and not aa[k].left_censored and not bb[k].left_censored]
            for population, left, right in (("full", aa.values(), bb.values()),
                                             ("matched", [aa[k] for k in common], [bb[k] for k in common]),
                                             ("completed_in_both", [aa[k] for k in completed], [bb[k] for k in completed])):
                ar, br = summarize(left), summarize(right)
                table = {}
                for metric in ar:
                    av, bv = ar[metric], br[metric]
                    table[metric] = {"baseline": av, "advance": bv,
                                     "absolute_difference": bv - av if av is not None and bv is not None else None,
                                     "percent_difference": 100 * (bv - av) / av if av not in (None, 0) and bv is not None else None}
                tables[population][tls][group] = table
    return tables


def fmt(value, signed=False):
    if value is None:
        return "N/A"
    if abs(value - round(value)) < 1e-10:
        return f"{value:+.0f}" if signed else f"{value:.0f}"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def write_report(output, tables, checks):
    labels = {"priority": "Corredor recto, todos los orígenes", "secondary": "Otros movimientos/secundario",
              "priority_j0": "Corredor recto liberado por J0", "priority_direct_e1": "Corredor recto insertado en E1",
              "priority_other_upstream": "Corredor recto de otros orígenes", "receiver_access_all": "Acceso receptor completo",
              "receiver_j0_all_movements": "Acceso receptor liberado por J0, incluidos giros",
              "receiver_turns_or_terminal": "Giros/fin de ruta del acceso receptor", "other_accesses": "Demás accesos"}
    definitions = ["# Auditoría diagnóstica C1: movimiento, origen y población", "",
        "El controlador está congelado. Estos datos no deciden si esta prioridad es la adecuada para la tesis.", "",
        "## Definiciones reproducibles", "",
        "- Corredor recto: conexión E1→E5 en J2, E5→E10 en J10 y E10→E13 en J16. Se usa ruta e índice actuales, incluso en carriles compartidos con giros.",
        "- Secundario: complemento de esos movimientos en los carriles controlados de cada slave. Incluye giros del acceso receptor y tráfico del boulevard en sentido contrario; no equivale exclusivamente a calles transversales.",
        "- J0: vehículo registrado causalmente por Fase B al entrar en E1 tras J0, con prefijo de ruta continuo E1 / E1,E5 / E1,E5,E10 hasta el TLS respectivo. Inserciones directas en E1 y entradas laterales se presentan aparte.",
        "- Acceso receptor completo: E1/E5/E10 con todos sus movimientos. Se presenta además el subconjunto de origen J0, incluidos giros de J16, sin asumir que E13 es la única definición posible de prioridad.",
        "- Waiting: segundos muestreados con velocidad <0.1 m/s en el acceso controlado. Se reconcilia exactamente con waiting_J2/J10/J16. No se suma getWaitingTime acumulativo ni waiting de vías externas.",
        "- Detención: entrada en un episodio de velocidad <0.1 m/s. Una parada sostenida cuenta una vez; cambiar de carril no reinicia el episodio. Una primera observación ya detenida cuenta una parada observada.",
        "- Vehículos: IDs distintos observados en el acceso durante 2000 s; waiting medio usa todos esos IDs, incluidos los que aún no cruzan. El agregado ALL suma waiting/paradas de los tres TLS y usa IDs únicos; throughput agregado cuenta IDs distintos que cruzan al menos un TLS, no salidas de la red ni necesariamente viajes completos.",
        "- Throughput local: cruces físicos observados al entrar en un carril interno del TLS o en el edge de salida previsto. Teleports/desapariciones no se consideran cruces. Cruce sin parada usa solo aproximaciones completas y excluye observaciones de inicio censuradas.",
        "- Tiempo de recorrido: detección post-J0 en E1 hasta cruce del TLS; solo para origen J0 con cruce completo. Para J16 representa el recorrido completo hasta esa intersección. También se entrega tiempo desde primera observación del acceso hasta cruce.",
        "- Remanente: miembro local válido único de un platoón cerrado originalmente múltiple, según la vista causal de C0 en ese paso. Es un subconjunto temporal, no una población adicional disjunta.",
        "- Población completa: todos los observados de cada corrida. Pareada: mismos (TLS, ID, índice de ruta) presentes en ambos y pertenecientes al mismo grupo. Completada en ambas: subconjunto pareado que cruzó en ambas, para comparar cruces sin parada y tiempos con igual denominador.",
        "- Horizonte: 2000 s, SUMO 1.26.0, demanda oficial, checkpoint model_future_v39_yellow_test36_3, PPO 5–45 s. Espera y paradas se truncan al horizonte; pares pueden tener tiempos de exposición distintos.", "",
        "## Validación de congelación", "",
        "OFF, SHADOW y ADVANCE reproducen exactamente sus evaluaciones anteriores: acciones, observaciones, reward, timings, señales, métricas y comandos. JSONL de coordinación y Fase B permanecen idénticos. OFF y ADVANCE se repiten; SHADOW coincide también con la nueva población OFF.",
        "Los hashes de archivos de control/checkpoint/red/demanda permanecen idénticos a los de inicio. La telemetría recibe una interfaz que solo expone getters; no puede escribir ni llamar simulationStep. Exactamente 2000 muestras por corrida.", ""]
    lines = definitions
    main_metrics = [("priority", "waiting", "Waiting prioritario"), ("secondary", "waiting", "Waiting secundario"),
                    ("priority", "vehicles", "Vehículos prioritarios"), ("secondary", "vehicles", "Vehículos secundarios"),
                    ("priority", "mean_waiting_per_vehicle", "Waiting medio prioritario"),
                    ("secondary", "mean_waiting_per_vehicle", "Waiting medio secundario"),
                    ("priority", "stops", "Detenciones prioritarias"), ("secondary", "stops", "Detenciones secundarias"),
                    ("priority", "no_stop_crossing_percent", "Cruces prioritarios sin parada (%)"),
                    ("secondary", "no_stop_crossing_percent", "Cruces secundarios sin parada (%)"),
                    ("priority", "mean_j0_to_crossing_seconds", "Tiempo J0→TLS prioritario (s)"),
                    ("priority", "throughput", "Throughput prioritario local"), ("secondary", "throughput", "Throughput secundario local")]
    for population, title in (("full", "Población completa"), ("matched", "Mismos vehículos presentes en ambas condiciones"),
                               ("completed_in_both", "Mismos vehículos que cruzan en ambas condiciones")):
        lines.extend([f"## {title}", "", "| TLS | Métrica | Baseline | Advance | Dif. abs | Dif. % |",
                      "|---|---|---:|---:|---:|---:|"])
        for tls in CORRIDOR_MOVEMENTS:
            for group, metric, label in main_metrics:
                row = tables[population][tls][group][metric]
                percent = fmt(row["percent_difference"], True)
                lines.append(f'| {tls} | {label} | {fmt(row["baseline"])} | {fmt(row["advance"])} | {fmt(row["absolute_difference"], True)} | {percent + "%" if percent != "N/A" else percent} |')
        for metric, label in (("waiting", "Waiting agregado"), ("stops", "Detenciones agregadas")):
            lines.extend(["", f"### {label}", "", "| Grupo | Baseline | Advance | Dif. % |", "|---|---:|---:|---:|"])
            for group in GROUPS:
                row = tables[population]["ALL"][group][metric]
                percent = fmt(row["percent_difference"], True)
                lines.append(f'| {labels[group]} | {fmt(row["baseline"])} | {fmt(row["advance"])} | {percent + "%" if percent != "N/A" else percent} |')
    lines.extend(["", "## Orígenes, acceso receptor y remanentes por TLS", "",
                  "Todas las métricas de todos los grupos y poblaciones están en comparison.json. Datos por vehículo en *.flow_metrics.json y eventos de entrada/parada/cruce en *.flow_events.jsonl.", "",
                  "| TLS | Grupo | Waiting base | Waiting adv | Paradas base | Paradas adv | Vehículos base | Vehículos adv |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for tls in CORRIDOR_MOVEMENTS:
        for group in GROUPS[2:]:
            rows = tables["full"][tls][group]
            lines.append(f'| {tls} | {labels[group]} | {fmt(rows["waiting"]["baseline"])} | {fmt(rows["waiting"]["advance"])} | {fmt(rows["stops"]["baseline"])} | {fmt(rows["stops"]["advance"])} | {fmt(rows["vehicles"]["baseline"])} | {fmt(rows["vehicles"]["advance"])} |')
    lines.extend(["", "| TLS | Remanentes prioritarios observados base/adv | Waiting como remanente base/adv | Paradas iniciadas como remanente base/adv |",
                  "|---|---:|---:|---:|"])
    for tls in CORRIDOR_MOVEMENTS:
        rows = tables["full"][tls]["priority_j0"]
        values = [f'{fmt(rows[m]["baseline"])}/{fmt(rows[m]["advance"])}' for m in ("remanent_observed_vehicles", "remanent_waiting", "stops_started_as_remanent")]
        lines.append(f'| {tls} | ' + ' | '.join(values) + ' |')
    lines.extend(["", "## Evidencia", "", "validation_summary.json contiene hashes, comparaciones exactas y repeticiones deterministas. Los ceros de una población vacía no se interpretan como mejora; medias y porcentajes sin denominador se muestran N/A.",
                  "No se emite un juicio sobre el objetivo de tesis ni se modifica el controlador. No se hizo commit ni push."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "logs/c1_diagnostic"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = frozen_hashes()
    cases = (("baseline", "off", ROOT / "logs/c1/baseline.evaluation.json"),
             ("baseline_repeat", "off", ROOT / "logs/c1/baseline.evaluation.json"),
             ("shadow", "shadow", ROOT / "logs/c1/c0_regression/local_shadow.evaluation.json"),
             ("advance", "advance", ROOT / "logs/c1/advance.evaluation.json"),
             ("advance_repeat", "advance", ROOT / "logs/c1/advance.evaluation.json"))
    results, diagnostics, checks = {}, {}, {}
    for name, mode, reference in cases:
        result, diagnostic = run(name, mode, output)
        historical = json.loads(reference.read_text())
        assert all(result[key] == value for key, value in historical.items()), name
        assert sumo_version(output / (name + ".tripinfo.xml")) == "1.26.0"
        historical_name = "local_shadow" if mode == "shadow" else "baseline" if mode == "off" else "advance"
        historical_folder = reference.parent
        for suffix in ("coordination.jsonl", "baseline_forecast.jsonl"):
            assert (output / (name + "." + suffix)).read_bytes() == (historical_folder / (historical_name + "." + suffix)).read_bytes(), (name, suffix)
        checks[name] = {"historical_evaluation_exact": True, "existing_jsonl_byte_identical": True,
                        "new_write_commands": 0, "telemetry_steps": diagnostic["steps"]}
        if mode == "advance":
            checks[name]["C1_safety"] = audit(result, output / (name + ".coordination.jsonl"))
        if name == "shadow":
            assert diagnostic == diagnostics["baseline"]
        if name.endswith("_repeat"):
            first = name.removesuffix("_repeat")
            assert result == results[first] and diagnostic == diagnostics[first]
            assert (output / (name + ".flow_events.jsonl")).read_bytes() == (output / (first + ".flow_events.jsonl")).read_bytes()
            checks[name]["deterministic_exact"] = True
        results[name], diagnostics[name] = result, diagnostic
        assert frozen_hashes() == frozen
        print(name, "frozen behavior exact; telemetry waiting reconciled", flush=True)
    tables = compare(diagnostics["baseline"], diagnostics["advance"])
    write_json(output / "comparison.json", tables)
    write_json(output / "validation_summary.json", {"frozen_hashes": frozen, "checks": checks,
                                                   "telemetry_sha256": digest(ROOT / "policy/flow_metrics.py")})
    write_report(output, tables, checks)
    print(json.dumps({group: {metric: tables["full"]["ALL"][group][metric] for metric in ("waiting", "stops", "vehicles")}
                      for group in ("priority", "secondary", "priority_j0")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
