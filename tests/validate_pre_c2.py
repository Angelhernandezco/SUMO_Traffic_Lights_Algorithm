"""Explicit full-horizon pre-C2 regression and read-only telemetry capture."""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import traci
from policy import train
from policy.corridor_telemetry import CorridorTelemetry, ReadOnlyTraCI
from validate_c1 import run
from validate_c0 import digest, write_json


def stream(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def capture(name, mode, destination):
    observer = CorridorTelemetry()
    sources = []
    view = ReadOnlyTraCI(traci)
    original = train.SumoTrafficEnv._record_step_metrics

    def observe(env):
        result = original(env)
        observer.observe(view, env.forecast)
        if not sources:
            sources.append(env.forecast)
        return result

    # validate_c1 caches completed controller runs. An observer must see every
    # step itself, so always force execution instead of reusing that cache.
    completion = destination / f"{name}.complete.json"
    if completion.exists():
        completion.unlink()
    with mock.patch.object(train.SumoTrafficEnv, "_record_step_metrics", autospec=True, side_effect=observe):
        evaluation = run(name, {"eta_mode": "local", "coordination_mode": mode}, destination)
    report = observer.export(sources[0])
    assert report["steps"] == 2000 and report["last_time"] == 2000
    prefix = destination / name
    stream(prefix.with_suffix(".vehicle_steps.jsonl"), observer.samples)
    stream(prefix.with_suffix(".vehicle_events.jsonl"), observer.events)
    stream(prefix.with_suffix(".platoon_steps.jsonl"), observer.platoon_steps)
    write_json(prefix.with_suffix(".corridor.json"), report)
    return evaluation, report


def event_rows(path, kind):
    return [row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if row["event"] == kind]


def compare(before, destination, after, name, before_name):
    old = json.loads((before / f"{before_name}.evaluation.json").read_text(encoding="utf-8"))
    fields = ("actions", "observations", "controls", "signals", "set_commands", "coordination_writes",
              "metrics", "supplementary")
    exact = {field: old[field] == after[field] for field in fields}
    exact["forecast_jsonl"] = (before / f"{before_name}.baseline_forecast.jsonl").read_bytes() == (
        destination / f"{name}.baseline_forecast.jsonl").read_bytes()
    old_stream = list(map(json.loads, (before / f"{before_name}.coordination.jsonl").read_text().splitlines()))
    new_stream = list(map(json.loads, (destination / f"{name}.coordination.jsonl").read_text().splitlines()))
    exact["existing_coordination_fields"] = len(old_stream) == len(new_stream) and all(
        all(right.get(key) == value for key, value in left.items()) for left, right in zip(old_stream, new_stream))
    old_events = event_rows(before / f"{before_name}.coordination.jsonl", "c1_execution")
    new_events = event_rows(destination / f"{name}.coordination.jsonl", "c1_execution")
    changed_interventions = [{"index": i, "before": left, "after": right}
                             for i, (left, right) in enumerate(zip(old_events, new_events))
                             if any(left.get(key) != right.get(key) for key in
                                    ("result", "reason", "effective_reduction", "remaining_after"))]
    return {"exact": exact, "old_executions": len(old_events), "new_executions": len(new_events),
            "changed_interventions": changed_interventions,
            "new_failures": [r for r in new_events if r["reason"] in
                             ("effective_opening_outside_horizon", "write_failed", "write_not_applied")]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=ROOT / "logs/pre_c2/before")
    parser.add_argument("--after", type=Path, default=ROOT / "logs/pre_c2/after")
    args = parser.parse_args()
    args.after.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, mode in (("off", "off"), ("shadow", "shadow"), ("advance", "advance"),
                       ("advance_repeat", "advance")):
        evaluation, report = capture(name, mode, args.after)
        result[name] = compare(args.before, args.after, evaluation, name,
                               "advance" if name == "advance_repeat" else name)
        result[name]["telemetry_steps"] = report["steps"]
        result[name]["telemetry_sha256"] = digest(args.after / f"{name}.vehicle_steps.jsonl")
        if name in ("off", "shadow"):
            assert all(result[name]["exact"].values()), name
        if name == "advance_repeat":
            for field in ("actions", "observations", "controls", "signals", "set_commands",
                          "coordination_writes", "metrics", "supplementary"):
                assert evaluation[field] == json.loads((args.after / "advance.evaluation.json").read_text())[field]
            assert digest(args.after / "advance.vehicle_steps.jsonl") == result[name]["telemetry_sha256"]
    write_json(args.after / "validation.json", result)
    print(json.dumps({name: {"exact": info["exact"], "changed_interventions": len(info["changed_interventions"])}
                      for name, info in result.items()}, indent=2))


if __name__ == "__main__":
    main()
