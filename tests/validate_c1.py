"""C1 experiment and full TraCI safety/determinism audit; explicit opt-in run."""

import argparse
import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import traci
from sumolib import checkBinary
from policy.agent import PPOAgent
from policy import train
from policy.forecast import CORRIDOR
from policy.coordination import AdvanceExecutor, NOMINAL_DURATIONS, TLS_ORDER
from validate_c0 import MODEL, EXPECTED, plain, digest, write_json, sumo_version, run as run_c0

def run(name, options, destination):
    # Capture changes invalidate evidence; audit/report edits recheck it.
    capture = inspect.getsource(run).split("    signals, actions", 1)[1].rstrip()
    signature = {"options": options, "checkpoint": digest(MODEL),
                 "capture": hashlib.sha256(capture.encode()).hexdigest(),
                 "code": {str(p.relative_to(ROOT)): digest(p) for p in
                          (ROOT / "policy/train.py", ROOT / "policy/coordination.py",
                           ROOT / "policy/forecast.py", ROOT / "policy/agent.py")}}
    completion = destination / (name + ".complete.json")
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["signature"] == signature and all(
                (destination / filename).exists() and digest(destination / filename) == expected
                for filename, expected in saved["outputs"].items()):
            print(name, "reusing complete matching evidence", flush=True)
            return json.loads((destination / (name + ".evaluation.json")).read_text())
    signals, actions, controls, observations, set_commands = [], [], [], [], []
    coordination_writes = []
    coordinating = [False]
    transverse = Counter()
    stopped_previous = set()
    corridor_stops = 0
    traci.start([checkBinary("sumo"), "-c", str(ROOT / "configuration.sumocfg"),
                 "--no-step-log", "true", "--duration-log.disable", "true",
                 "--tripinfo-output", str(destination / (name + ".tripinfo.xml"))])
    try:
        contexts = train._build_intersection_contexts()
        env = train._make_env(contexts=contexts, min_green=5, max_green=45, **options)
        agent = PPOAgent(32, action_dim=1, hidden_dim=256, lr=2e-4,
                         gamma=0.99, gae_lambda=0.95, clip_eps=0.15,
                         entropy_coef=0.07, value_coef=0.5, max_grad_norm=0.35,
                         ppo_epochs=4, minibatch_size=64, normalize_obs=True,
                         concentration_floor=0.2, init_action_mean=0.095,
                         init_total_concentration=3.0, device=torch.device("cpu"))
        metadata = agent.load(str(MODEL), map_location=torch.device("cpu"))
        assert metadata["min_green"] == 5 and metadata["max_green"] == 45
        rms_before = plain({"mean": agent.obs_rms.mean, "var": agent.obs_rms.var,
                            "count": agent.obs_rms.count})
        original_step, original_act = env.step, agent.act
        original_simulation_step = traci.simulationStep
        original_set = traci.trafficlight._setCmd

        def audit_set(var_id, tls, format="", *values):
            entry = {"time": traci.simulation.getTime(), "tls": tls,
                     "variable": var_id, "format": format, "values": plain(values)}
            set_commands.append(entry)
            if coordinating[0]:
                assert tls in train.SLAVE_TLS_IDS, entry
                assert var_id == traci.constants.TL_PHASE_DURATION, entry
                coordination_writes.append(entry)
            elif tls in train.SLAVE_TLS_IDS:
                raise AssertionError(entry)
            if tls in train.SLAVE_TLS_IDS:
                assert options.get("coordination_mode") == "advance", entry
            return original_set(var_id, tls, format, *values)

        def capture_simulation(*args, **kwargs):
            nonlocal corridor_stops, stopped_previous
            result = original_simulation_step(*args, **kwargs)
            for tls, edge in CORRIDOR:
                transverse[tls] += sum(traci.lane.getLastStepHaltingNumber(lane)
                                       for lane in contexts[tls].controlled_lanes
                                       if lane.rsplit("_", 1)[0] != edge)
            stopped = {vid for _, edge in CORRIDOR
                       for vid in traci.edge.getLastStepVehicleIDs(edge)
                       if traci.vehicle.getSpeed(vid) < 0.1}
            corridor_stops += len(stopped - stopped_previous)
            stopped_previous = stopped
            signals.append({"time": traci.simulation.getTime(), "signals": {
                tls: {"program": traci.trafficlight.getProgram(tls),
                      "phase": traci.trafficlight.getPhase(tls),
                      "state": traci.trafficlight.getRedYellowGreenState(tls),
                      "next_switch": traci.trafficlight.getNextSwitch(tls),
                      "duration": traci.trafficlight.getPhaseDuration(tls),
                      "spent_duration": traci.trafficlight.getSpentDuration(tls)}
                for tls in contexts}})
            return result

        def capture_act(state, **kwargs):
            observations.append(plain(state))
            return original_act(state, **kwargs)

        def capture_step(action, **kwargs):
            time = traci.simulation.getTime()
            cursor = env.phase_cursor
            result = original_step(action, **kwargs)
            observation, reward, done, duration, waiting, info = result
            actions.append(plain({"time": time, "end_time": traci.simulation.getTime(),
                                  "phase_cursor": cursor,
                                  "phase_index": env.phases[cursor]["index"],
                                  "action": action, "next_observation": observation,
                                  "reward": reward, "waiting": waiting, "info": info}))
            return result

        def capture_control(method):
            original = getattr(traci.trafficlight, method)

            def call(*args, **kwargs):
                controls.append(plain({"time": traci.simulation.getTime(),
                                       "method": method, "args": args, "kwargs": kwargs}))
                return original(*args, **kwargs)
            return call

        def capture_execute(*args, **kwargs):
            coordinating[0] = True
            try:
                return original_execute(*args, **kwargs)
            finally:
                coordinating[0] = False

        from contextlib import ExitStack
        with ExitStack() as stack:
            if options.get("coordination_mode") == "advance":
                original_execute = AdvanceExecutor.execute
                stack.enter_context(mock.patch.object(AdvanceExecutor, "execute", autospec=True,
                                                      side_effect=capture_execute))
            stack.enter_context(mock.patch.object(traci.trafficlight, "_setCmd", side_effect=audit_set))
            stack.enter_context(mock.patch.object(traci, "simulationStep", side_effect=capture_simulation))
            stack.enter_context(mock.patch.object(env, "step", side_effect=capture_step))
            stack.enter_context(mock.patch.object(agent, "act", side_effect=capture_act))
            for method in ("setRedYellowGreenState", "setPhase", "setPhaseDuration",
                           "setProgram", "setProgramLogic"):
                stack.enter_context(mock.patch.object(traci.trafficlight, method,
                                                      side_effect=capture_control(method)))
            metrics = train._run_single_episode(agent=agent, env=env, steps=2000,
                                                collect_rollout=False, deterministic=True,
                                                debug=False, debug_limit=0, update_obs_rms=False)

        assert rms_before == plain({"mean": agent.obs_rms.mean, "var": agent.obs_rms.var,
                                    "count": agent.obs_rms.count})
        if options.get("coordination_mode") != "advance":
            for key, value in EXPECTED.items():
                assert metrics[key] == value, (name, key, metrics[key], value)
        cohort = [v for v in env.forecast.followed.values() if v.route_class == "e1_e5_e10"
                  and v.status_by_slave["J16"] != "no_aplicable"]
        completed = {v.vehicle_id: v.actual_by_slave["J16"] - v.detected_at for v in cohort
                     if "J16" in v.actual_by_slave and v.censor_reason is None}
        supplementary = {
            "waiting_transverse_slaves": dict(transverse),
            "waiting_transverse_slaves_total": sum(transverse.values()),
            "corridor_stops": corridor_stops,
            "corridor_completed": len(completed), "corridor_tracked": len(cohort),
            "corridor_incomplete_or_censored": len(cohort) - len(completed),
            "corridor_mean_seconds": sum(completed.values()) / len(completed) if completed else None,
            "corridor_times_by_vehicle": completed,
        }
        result = plain({"metrics": metrics, "actions": actions, "controls": controls,
                        "observations": observations, "signals": signals,
                        "set_commands": set_commands, "coordination_writes": coordination_writes,
                        "supplementary": supplementary})
        write_json(destination / (name + ".evaluation.json"), result)
        if env.forecast is not None:
            env.forecast.finalize(float(traci.simulation.getTime()))
            env.forecast.store.write_jsonl(destination / (name + ".baseline_forecast.jsonl"))
        if getattr(env, "c0", None) is not None:
            env.c0.finalize(float(traci.simulation.getTime()))
            env.c0.write_jsonl(destination / (name + ".coordination.jsonl"))
            write_json(destination / (name + ".c0_summary.json"), env.c0.summary())
        outputs = [name + ".evaluation.json", name + ".tripinfo.xml"]
        if env.forecast is not None:
            outputs.append(name + ".baseline_forecast.jsonl")
        if env.c0 is not None:
            outputs.extend((name + ".coordination.jsonl", name + ".c0_summary.json"))
        print(name, "metrics", {k: metrics[k] for k in EXPECTED}, "slave writes", len(coordination_writes), flush=True)
        return result
    finally:
        traci.close()
        # SUMO closes tripinfo only when the connection closes.
        if "outputs" in locals():
            write_json(completion, {"signature": signature,
                                    "outputs": {f: digest(destination / f) for f in outputs}})


def events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit(result, path):
    stream = events(path)
    configured = next(e["programs"] for e in stream if e["event"] == "c0_configuration")
    assert len(result["signals"]) == 2000
    assert [s["time"] for s in result["signals"]] == list(range(1, 2001))
    completed_phases = Counter()
    previous = {}
    for sample in result["signals"]:
        for tls in TLS_ORDER:
            signal = sample["signals"][tls]
            phase = signal["phase"]
            start = sample["time"] - signal["spent_duration"]
            assert signal["program"] == configured[tls]["id"]
            assert signal["state"] == configured[tls]["states"][phase]
            assert signal["duration"] == NOMINAL_DURATIONS[tls][phase]
            if tls in previous:
                old_phase, old_start, old_state = previous[tls]
                if old_phase != phase:
                    assert phase == (old_phase + 1) % len(NOMINAL_DURATIONS[tls])
                    elapsed = start - old_start
                    if old_phase % 2 == 1 or old_phase == 2:
                        assert elapsed == NOMINAL_DURATIONS[tls][old_phase], (tls, old_phase, elapsed)
                    else:
                        minimum = 36 if tls == "J10" else 10
                        assert minimum <= elapsed <= NOMINAL_DURATIONS[tls][old_phase], (tls, old_phase, elapsed)
                    completed_phases[tls] += 1
                else:
                    assert start == old_start and signal["state"] == old_state
            previous[tls] = (phase, start, signal["state"])
    executions = [e for e in stream if e["event"] == "c1_execution"]
    reservations = [e for e in stream if e["event"] == "shadow_decision" and e["result"] == "shadow_reserved"]
    assert len(executions) == len(reservations)
    reserved_keys = {(e["time"], e["tls_id"], e["platoon_id"], e["receptor_occurrence"], e["revision"])
                     for e in reservations}
    successful = [e for e in executions if e["result"] == "executed"]
    seen = set()
    last_occurrence = {}
    for e in executions:
        assert (e["time"], e["tls_id"], e["platoon_id"], e["receptor_occurrence"],
                e["reservation_revision"]) in reserved_keys
    for e in successful:
        tls, occurrence = e["tls_id"], e["receptor_occurrence"]
        key = (tls, occurrence)
        assert key not in seen
        seen.add(key)
        assert occurrence >= last_occurrence.get(tls, -1) + 2
        last_occurrence[tls] = occurrence
        assert e["phase"] % 2 == 0 and e["phase"] != 2 and "y" not in e["state"].lower()
        assert 0 <= e["time"] - e["observed_at"] <= 1
        assert e["time"] + e["remaining_after"] - e["phase_started"] >= e["minimum_green"]
        assert 1 <= e["effective_reduction"] <= 5
        assert e["remaining_after"] >= 1
        assert e["remaining_before"] - e["remaining_after"] == e["effective_reduction"]
        assert e["budget_used_before"] == 0 and not e["cooldown"]
    slave_writes = [c for c in result["set_commands"] if c["tls"] in TLS_ORDER]
    assert slave_writes == result["coordination_writes"]
    assert len(slave_writes) == len(successful)
    for command, e in zip(slave_writes, successful):
        assert command["variable"] == traci.constants.TL_PHASE_DURATION
        assert (command["time"], command["tls"], command["values"]) == (
            e["time"], e["tls_id"], [e["remaining_after"]])
    assert all(c["method"] == "setPhaseDuration" for c in result["controls"] if c["args"][0] in TLS_ORDER)
    return {"completed_phases_checked": dict(completed_phases), "slave_writes": len(slave_writes),
            "coordination_writes_to_J0": 0, "extra_simulation_steps": 0,
            "unexecuted_reservations": len(executions) - len(successful)}


def comparison(baseline, advance):
    values = {}
    for key in EXPECTED:
        a, b = baseline["metrics"][key], advance["metrics"][key]
        values[key] = {"baseline": a, "advance": b, "absolute_difference": b - a,
                       "percent_difference": 100 * (b - a) / a if a else None}
    for key in ("waiting_transverse_slaves_total", "corridor_stops", "corridor_mean_seconds",
                "corridor_completed", "corridor_incomplete_or_censored"):
        a, b = baseline["supplementary"][key], advance["supplementary"][key]
        values[key] = {"baseline": a, "advance": b, "absolute_difference": b - a,
                       "percent_difference": 100 * (b - a) / a if a else None}
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "logs/c1"))
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    protected = json.loads((ROOT / "logs/c0/protected_hashes.json").read_text())
    assert {p: digest(ROOT / p) for p in protected} == protected
    baseline = run("baseline", {"eta_mode": "local"}, destination)
    historical = json.loads((ROOT / "logs/c0/local_off.evaluation.json").read_text())
    assert all(baseline[k] == v for k, v in historical.items())
    assert (destination / "baseline.coordination.jsonl").read_bytes() == (
        ROOT / "logs/c0/local_off.coordination.jsonl").read_bytes()
    regression = destination / "c0_regression"
    regression.mkdir(parents=True, exist_ok=True)
    shadow_checks = {}
    for name in ("local_shadow", "local_shadow_repeat1", "local_shadow_repeat2"):
        result = run_c0(name, {"eta_mode": "local", "coordination_mode": "shadow"}, regression)
        reference = json.loads((ROOT / "logs/c0" / (name + ".evaluation.json")).read_text())
        assert result == reference
        assert sumo_version(regression / (name + ".tripinfo.xml")) == "1.26.0"
        for suffix in ("coordination.jsonl", "baseline_forecast.jsonl", "c0_summary.json"):
            assert (regression / (name + "." + suffix)).read_bytes() == (
                ROOT / "logs/c0" / (name + "." + suffix)).read_bytes()
        c0_summary = json.loads((regression / (name + ".c0_summary.json")).read_text())
        assert [c0_summary["shadow"][tls]["geometric_pairs"] for tls in TLS_ORDER] == [22, 9, 14]
        assert [c0_summary["shadow"][tls]["shadow_reservations"] for tls in TLS_ORDER] == [8, 5, 8]
        shadow_checks[name] = {"evaluation_exact": True, "jsonl_exact_to_C0": True,
                               "coordination_sha256": digest(regression / (name + ".coordination.jsonl"))}
    results, checks = [], {}
    for name in ("advance", "advance_repeat1", "advance_repeat2"):
        result = run(name, {"eta_mode": "local", "coordination_mode": "advance"}, destination)
        assert sumo_version(destination / (name + ".tripinfo.xml")) == "1.26.0"
        checks[name] = audit(result, destination / (name + ".coordination.jsonl"))
        if results:
            assert result == results[0], name
            for suffix in ("coordination.jsonl", "baseline_forecast.jsonl"):
                assert (destination / (name + "." + suffix)).read_bytes() == (
                    destination / ("advance." + suffix)).read_bytes(), (name, suffix)
        results.append(result)
    assert sumo_version(destination / "baseline.tripinfo.xml") == "1.26.0"
    assert {p: digest(ROOT / p) for p in protected} == protected
    summary = json.loads((destination / "advance.c0_summary.json").read_text())
    common = baseline["supplementary"]["corridor_times_by_vehicle"].keys() & results[0]["supplementary"]["corridor_times_by_vehicle"].keys()
    paired = {"completed_in_both": len(common),
              "baseline_mean_seconds": sum(baseline["supplementary"]["corridor_times_by_vehicle"][v] for v in common) / len(common),
              "advance_mean_seconds": sum(results[0]["supplementary"]["corridor_times_by_vehicle"][v] for v in common) / len(common)} if common else None
    manifest = {"checks": checks, "shadow_checks": shadow_checks,
                "deterministic_repeats": 3, "protected_hashes": protected,
                "sumo_version": "1.26.0", "comparison": comparison(baseline, results[0]),
                "advance_summary": summary, "paired_corridor": paired,
                "supplementary_definitions": {
                    "waiting_transverse_slaves": "halting vehicle-seconds on slave controlled lanes excluding E1/E5/E10 incoming edges",
                    "corridor_stops": "entries into speed <0.1 m/s on E1/E5/E10; new observations already stopped count once",
                    "corridor_mean_seconds": "E1 detection through observed J16 crossing, completed applicable original master cohort only",
                    "waiting_total_network": "union of controlled incoming lanes at the four TLS; not every road",
                }}
    write_json(destination / "validation_summary.json", manifest)
    lines = ["# C1: baseline local/off vs local/advance", "",
             "Checkpoint model_future_v39_yellow_test36_3, demanda oficial, SUMO 1.26.0, 2000 s, PPO 5–45 s.", "",
             "| Métrica | Baseline | Advance | Diferencia absoluta | Diferencia % |",
             "|---|---:|---:|---:|---:|"]
    for key, row in manifest["comparison"].items():
        percent = f'{row["percent_difference"]:+.2f}%' if row["percent_difference"] is not None else "N/A"
        lines.append(f'| {key} | {row["baseline"]:.2f} | {row["advance"]:.2f} | {row["absolute_difference"]:+.2f} | {percent} |')
    lines.extend(["", "| TLS | Intervenciones reales | Segundos reducidos | Reservas no ejecutadas |",
                  "|---|---:|---:|---:|"])
    for tls in TLS_ORDER:
        row = summary["advance"][tls]
        lines.append(f'| {tls} | {row["real_interventions"]} | {row["seconds_reduced"]:g} | {row["unexecuted_reservations"]} |')
    lines.extend(["", "Tres corridas ADVANCE idénticas, incluidos eventos, acciones, observaciones, reward, señales y métricas.",
                  "OFF y tres corridas SHADOW reproducen C0 exactamente, incluidos JSONL, candidatos 22/9/14 y reservas 8/5/8.",
                  "La auditoría comprueba exclusivamente setPhaseDuration en slaves, cero escrituras de coordinación a J0, secuencia cíclica, mínimos, máximo de 5 s, una intervención por apertura y descanso.",
                  "Amarillos y verdes receptores conservan su duración completa, incluido amarillo J2 de 24 s. Se ejecutan exactamente 2000 simulationStep.", "",
                  "Las métricas suplementarias son observacionales y no entran en decisiones. El tiempo del corredor incluye solo recorridos completados; se registra también la censura y la comparación de vehículos comunes."])
    if paired:
        lines.extend(["", f'Vehículos que completan el corredor en ambas condiciones: {paired["completed_in_both"]}; '
                      f'media baseline {paired["baseline_mean_seconds"]:.2f} s; '
                      f'advance {paired["advance_mean_seconds"]:.2f} s.'])
    worse = manifest["comparison"]["waiting_total_network"]["absolute_difference"] >= 0
    if worse:
        lines.extend(["", "Interpretación: C1 no cumple el criterio de mejora de waiting total en este escenario. "
                      "Se detiene aquí el experimento, sin añadir extensión ni otras heurísticas. "
                      "La repetibilidad demuestra determinismo para esta demanda, no generalización a otras demandas."])
    lines.extend(["", "Sin cambios en PPO/checkpoint, acción Beta, reward/observaciones de J0, configuración, red ni demanda. "
                  "La política J0 responde a la nueva dinámica de tráfico en ADVANCE; no recibe escrituras de coordinación.",
                  "Las razones de reservas no ejecutadas se encuentran por TLS en advance_summary.advance.loss_reasons del JSON; un diccionario vacío significa cero pérdidas de ejecución."])
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": manifest["comparison"], "advance": summary["advance"],
                      "paired_corridor": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()

