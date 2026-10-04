"""Collect successful replay evidence and the compact 15-row comparison."""
import argparse
import hashlib
import json
from pathlib import Path

from sync.common import ROOT, CHECKPOINT
from sync.metrics import write_csv
from tests.validate_sync import REFERENCE, compare, protected


def compact(summary):
    s = summary
    full, straight = s["full_corridor_any_exit"], s["full_corridor_straight_exit"]
    return {
        "seed": s["seed"], "mode": s["mode"], "scope": "full",
        "no_stop_J2": s["no_stop"], "cohort_J2": s["cohort"], "no_stop_J2_pct": s["no_stop_pct"],
        "waiting_E1_s": s["waiting_E1_s"], "mean_E1_to_E5_s": s["mean_E1_to_E5_s"],
        "no_stop_full": full["no_stop_all_three"], "cohort_full": full["cohort"],
        "no_stop_full_pct": full["no_stop_all_three_pct"], "full_waiting_s": full["waiting_total_s"],
        "stops_per_vehicle": full["stops_per_vehicle"],
        "mean_J0_to_J16_s": full["J0_to_J16_s"].get("mean"),
        "median_J0_to_J16_s": full["J0_to_J16_s"].get("median"),
        "p90_J0_to_J16_s": full["J0_to_J16_s"].get("p90"),
        "secondary_J2_s": s["secondary_wait_J2_s"], "secondary_J10_s": s["secondary_wait_s"]["J10"],
        "secondary_J16_s": s["secondary_wait_s"]["J16"], "network_waiting_s": s["network_total_wait_s"],
        "waiting_time": s["waiting_time"], "effective_flow": s["effective_flow"],
        "avg_queue_length": s["avg_queue_length"], "arrived": s["arrived"],
        "pending": s["pending"]["min_expected"], "corridor_pending": full["pending"],
        "strict_no_stop": straight["no_stop_all_three"], "strict_cohort": straight["cohort"],
        "secondary_total_s": s["controlled_secondary_wait_s"],
        "pending_active": s["pending"]["active"], "pending_insertion": s["pending"]["pending_insertion"],
        "horizon_s": s["horizon_s"], "SUMO_seed": s["SUMO_seed"], "step_s": 1,
        "offsets": s["offsets"], "delays_s": s["cumulative_delays_s"],
        "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
        "green_min_s": 5, "green_max_s": 45, "yellow_s": 4, "deterministic": True,
        "route_sha256": s["route_sha256"], "constraint_violations": s["constraint_violations"],
        "teleports": s["teleported"], "reference_equal": True,
    }


def load_runs(paths, reference):
    results = {}
    evidence = []
    for path in paths:
        checks = json.loads((path / "validation.json").read_text(encoding="utf-8"))
        for check in checks:
            assert check["summary_equal"] and check["all_seven_traces_equal"]
            assert check["constraint_violations"] == 0
            key = check["run"]
            actual = json.loads((path / f"{key}.json").read_text(encoding="utf-8"))
            compare(actual, reference["runs"][key]["summary"])
            assert key not in results, f"Duplicate run: {key}"
            results[key] = actual
            evidence.append(check)
    return results, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--gui", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    protected(reference)
    results, evidence = load_runs(args.runs, reference)
    assert results.keys() == reference["runs"].keys()
    gui, gui_evidence = load_runs([args.gui], reference)
    assert set(gui) == {f"seed42_{mode}_full" for mode in ("v1", "v2", "v3")}
    for key in gui:
        compare(gui[key], results[key])
    native = {}
    for name, count in (("receivers", 9), ("copy", 7), ("preannounce", 6)):
        cases = json.loads((ROOT / f"results/mechanism_{name}.json").read_text(encoding="utf-8"))
        assert len(cases) == count
        native[name] = {"passed": count, "cases": [c.get("case", f"{c.get('tls')}/{c.get('mode')}/empty={c.get('empty')}") for c in cases]}
    smoke = json.loads((ROOT / "results/original_smoke.json").read_text(encoding="utf-8"))
    assert all(smoke[k] == "passed" for k in ("PPO", "DQN", "plain", "heuristic"))
    rows = [compact(results[key]) for key in sorted(results)]
    totals = {mode: sum(r["no_stop_full"] for r in rows if r["mode"] == mode) for mode in ("v1", "v2", "v3")}
    assert totals == {"v1": 51, "v2": 61, "v3": 84}
    output = REFERENCE.parent
    write_csv(output / "per_demand.csv", rows)
    validation = {
        "source_commit": reference["source_commit"], "environment": reference["environment"],
        "full_runs": evidence, "GUI_seed42": gui_evidence, "native_cases": native,
        "original_modes": smoke, "protected_hashes_unchanged": True,
        "summary_float_absolute_tolerance": 1e-12, "canonical_traces_exact": True,
        "no_stop_full_totals": totals, "full_cohort_total_per_mode": 93,
        "trace_export_equivalence": "GUI + CSV export and headless summaries match the same seven frozen signatures",
        "code_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [ROOT / "run_sync.py", *sorted((ROOT / "sync").glob("*.py"))]},
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print("PASS: 15 full runs, 3 GUI replays, 22 native cases, 4 original modes")


if __name__ == "__main__":
    main()
