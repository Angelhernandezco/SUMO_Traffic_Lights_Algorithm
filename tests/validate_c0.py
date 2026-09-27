"""Full checkpoint validation of C0; run explicitly, not during unit discovery."""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
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


MODEL = ROOT / "policy/models/model_future_v39_yellow_test36_3.pth"
EXPECTED = {"waiting_J0": 4222.0, "waiting_J2": 12613.0,
            "waiting_J10": 3766.0, "waiting_J16": 6264.0,
            "waiting_total_network": 26865.0, "throughput": 604}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(plain(value), sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def sumo_version(tripinfo):
    with tripinfo.open(encoding="utf-8") as stream:
        header = stream.read(1024)
    match = re.search(r"by Eclipse SUMO sumo ([\d.]+)", header)
    assert match is not None, tripinfo
    return match.group(1)


def opportunity_losses(path, summary):
    """Attribute lost unique geometric pairs, never repeated decision steps."""
    groups = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event"] == "shadow_decision" and event["geometric_candidate"]:
            key = (event["tls_id"], event["platoon_id"], event["receptor_occurrence"])
            groups[key].append(event)
    result = {}
    for tls in ("J2", "J10", "J16"):
        losses, details = Counter(), []
        reserved = 0
        for (target, pid, occurrence), events in sorted(groups.items()):
            if target != tls:
                continue
            if any(e["result"] == "shadow_reserved" for e in events):
                reserved += 1
                continue
            reasons = {r for e in events for r in e["failed_conditions"]}
            if all("opening_outside_horizon" in e["failed_conditions"] for e in events):
                reason = "opening_outside_horizon"
            elif "cooldown" in reasons:
                reason = "cooldown"
            else:
                assert reasons & {"budget_reserved", "conflict_selected_other"}, reasons
                reason = "budget_or_conflict"
            losses[reason] += 1
            details.append({"platoon_id": pid, "occurrence": occurrence,
                            "reason": reason, "all_reasons": sorted(reasons)})
        totals = summary["shadow"][tls]
        assert reserved == totals["shadow_reservations"]
        assert reserved + sum(losses.values()) == totals["geometric_pairs"]
        result[tls] = {"unique_lost_pairs": dict(sorted(losses.items())), "details": details}
    return result


def run(name, options, destination):
    signature = {"options": options, "checkpoint": digest(MODEL),
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
            assert tls not in train.SLAVE_TLS_IDS, entry
            return original_set(var_id, tls, format, *values)

        def capture_simulation(*args, **kwargs):
            result = original_simulation_step(*args, **kwargs)
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

        from contextlib import ExitStack
        with ExitStack() as stack:
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
        for key, value in EXPECTED.items():
            assert metrics[key] == value, (name, key, metrics[key], value)
        result = plain({"metrics": metrics, "actions": actions, "controls": controls,
                        "observations": observations, "signals": signals,
                        "set_commands": set_commands})
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
        print(name, "baseline exact; slave set commands=0", flush=True)
        return result
    finally:
        traci.close()
        # SUMO closes tripinfo only when the connection closes.
        if "outputs" in locals():
            write_json(completion, {"signature": signature,
                                    "outputs": {f: digest(destination / f) for f in outputs}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "logs/c0"))
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    protected = [MODEL, ROOT / "policy/agent.py", ROOT / "policy/forecast.py",
                 ROOT / "sumo_utils.py", ROOT / "configuration.sumocfg",
                 ROOT / "configuration.v39.sumocfg", ROOT / "maps/master_slave.net.xml",
                 ROOT / "maps/master_slave.rou.xml"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    if args.baseline_only:
        run("pre_c0", {}, destination)
        write_json(destination / "protected_hashes.json", hashes)
        return
    assert hashes == json.loads((destination / "protected_hashes.json").read_text())
    reference = json.loads((destination / "pre_c0.evaluation.json").read_text())
    historical = json.loads((ROOT / "logs/timeout_fix/master_on.evaluation.json").read_text())
    checks = {"recovered_baseline": {key: reference[key] == historical[key]
                                     for key in ("actions", "controls", "signals", "metrics")}}
    assert all(checks["recovered_baseline"].values())
    historical_off = json.loads((ROOT / "logs/timeout_fix/master_off.evaluation.json").read_text())
    checks["historical_off_on"] = {key: historical_off[key] == historical[key]
                                    for key in ("actions", "controls", "signals", "metrics")}
    assert all(checks["historical_off_on"].values())
    assert len(reference["signals"]) == 2000
    assert all(command["tls"] == "J0" for command in reference["set_commands"])
    cases = [("local_off", {"eta_mode": "local"}),
             ("local_shadow", {"eta_mode": "local", "coordination_mode": "shadow"}),
             ("local_shadow_repeat1", {"eta_mode": "local", "coordination_mode": "shadow"}),
             ("local_shadow_repeat2", {"eta_mode": "local", "coordination_mode": "shadow"})]
    for name, options in cases:
        result = run(name, options, destination)
        assert sumo_version(destination / (name + ".tripinfo.xml")) == "1.26.0"
        checks[name] = {key: result[key] == reference[key] for key in reference}
        assert all(checks[name].values()), (name, checks[name])
        for key in ("actions", "controls", "signals", "metrics"):
            assert result[key] == historical[key], (name, "historical", key)
    base_bytes = (destination / "local_shadow.coordination.jsonl").read_bytes()
    old_forecast = (ROOT / "logs/timeout_fix/forecast_validation.jsonl").read_bytes()
    for name, _ in cases:
        assert (destination / (name + ".baseline_forecast.jsonl")).read_bytes() == old_forecast
    checks["phase_b_events_preserved"] = {"jsonl_byte_equal": True}
    for name in ("local_shadow_repeat1", "local_shadow_repeat2"):
        assert (destination / (name + ".coordination.jsonl")).read_bytes() == base_bytes
        assert (destination / (name + ".baseline_forecast.jsonl")).read_bytes() == (
            destination / "local_shadow.baseline_forecast.jsonl").read_bytes()
    assert {str(p.relative_to(ROOT)): digest(p) for p in protected} == hashes
    summary = json.loads((destination / "local_shadow.c0_summary.json").read_text())
    losses = opportunity_losses(destination / "local_shadow.coordination.jsonl", summary)
    write_json(destination / "opportunity_losses.json", losses)
    manifest = {"checks": checks, "historical_exact": True,
                "slave_set_commands": 0, "shadow_jsonl_byte_equal": True,
                "shadow_sha256": hashlib.sha256(base_bytes).hexdigest(),
                "protected_hashes": hashes, "traci_version": getattr(traci, "__version__", None),
                "sumo_version": "1.26.0",
                "reused_evidence": ["pre_c0.evaluation.json", "logs/timeout_fix/master_off.evaluation.json",
                                    "logs/timeout_fix/master_on.evaluation.json",
                                    "logs/timeout_fix/forecast_validation.jsonl"],
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "opportunity_losses": losses, "summary": summary}
    write_json(destination / "validation_summary.json", manifest)
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
