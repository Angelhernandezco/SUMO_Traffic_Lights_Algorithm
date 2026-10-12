"""Reproducible matched 240-run experiment. Default stage only inspects the protocol."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from sync.common import CHECKPOINT, DEMAND, END, NET, OFFSETS, ROOT, SUMO_HOME, sumo_binary
from sync.metrics import route_ids, validate_phases, write_csv
from tests.collect_validation import compact
from tests.validate_sync import canonical, compare
from tools.compare_corridor import LABELS as HISTORICAL_LABELS, baseline, physical_coverage
from sync.simulation import run
from sync.v3 import PreannouncedWindow
from sync.v3_variants import WINDOW_CONTROLLERS
from run_sync import TRACE_NAMES

MODES = ("plain", "plain_offset", "only_ppo", "v3")
LABELS = dict(zip(MODES, ("Normal", "Normal sincronizado", "Sólo PPO", "PPO sincronizado")))
LEVELS = {"bajo": {"seeds": list(range(1001, 1031)), "period_s": 2.0},
          "alto": {"seeds": list(range(2001, 2031)), "period_s": 0.5}}
DEFAULT_OUTPUT = ROOT / "analysis/matched_240"
REFERENCE = ROOT / "analysis/corridor_full/reference.json"
SIX_CSV = ROOT / "analysis/six_configurations/per_demand.csv"
GENERATOR = ROOT / "tools/randomTrips.py"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scratch(prefix):
    # Windows sandbox temp permits writes but may reject replace/rename.
    # Keep atomic publications and duarouter validation in the writable workspace.
    parent = ROOT / "results"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=parent)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    write_csv(temporary, rows)
    if rows:
        os.replace(temporary, path)
    elif path.exists():
        path.unlink()  # Never leave a stale CSV suggesting a completed batch.


def route_content(path):
    """Canonical XML data without comments, dates or formatting (includes attributes)."""
    def node(element):
        return [element.tag, sorted(element.attrib.items()), (element.text or "").strip(),
                [node(child) for child in element]]
    return node(ET.parse(path).getroot())


def cohorts(path):
    j2 = route_ids(path, ["-E0", "E1", "E5"], END - 180)
    full, straight = set(), set()
    for vehicle in ET.parse(path).iter("vehicle"):
        if float(vehicle.get("depart")) > END - 180:
            continue
        edges = vehicle.find("route").get("edges").split()
        for i in range(len(edges) - 4):
            if edges[i:i + 4] == ["-E0", "E1", "E5", "E10"]:
                full.add(vehicle.get("id"))
                if edges[i + 4] == "E13":
                    straight.add(vehicle.get("id"))
    return {"J2": j2, "full": full, "strict": straight}


def percentage(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def historical_row(result):
    row = compact(result[0])
    mode = row["mode"]
    row.update(order=MODES.index(mode) + 1, label=HISTORICAL_LABELS[mode],
               reference_equal=True if mode == "v3" else None,
               checkpoint=None if mode.startswith("plain") else row["checkpoint"],
               green_min_s=15 if mode.startswith("plain") else 5,
               green_max_s=15 if mode.startswith("plain") else 45)
    for tls, coverage in physical_coverage(result).items():
        row[f"physical_window_coverage_{tls}_pct"] = coverage["coverage_pct"]
    return row


def measured_row(result, route):
    """Extend compact output using existing passive records, never controller decisions."""
    summary, _, _, _, timeline, _, phases, vehicle_rows = result
    row = historical_row(result)
    sets = cohorts(route)
    records = {r["vehicle_id"]: r for r in vehicle_rows}
    observed = [records[v] for v in sets["J2"]
                if v in records and records[v]["J2_entered_at"] is not None]
    full, straight = summary["full_corridor_any_exit"], summary["full_corridor_straight_exit"]
    row.update(
        label=LABELS[row["mode"]], reference_equal=None,
        scheduled_vehicles=sum(1 for _ in ET.parse(route).iter("vehicle")),
        inserted_vehicles=summary["departed"], completed_vehicles=summary["arrived"],
        completed_J2=summary["crossed"], pending_J2=summary["cohort"] - summary["crossed"],
        completed_full=full["completed"], completed_strict=straight["completed"],
        pending_strict=straight["pending"],
        no_stop_J2_completed_pct=percentage(row["no_stop_J2"], summary["crossed"]),
        no_stop_full_completed_pct=percentage(row["no_stop_full"], full["completed"]),
        completed_J2_pct=percentage(summary["crossed"], row["cohort_J2"]),
        completed_full_pct=percentage(full["completed"], row["cohort_full"]),
        cohort_J2_observed_E1=len(observed),
        waiting_E1_all_observed_s=sum(r["J2_waiting_s"] for r in observed) if observed else None,
        n_mean_E1_to_E5=summary["crossed"], n_J0_to_J16=full["J0_to_J16_s"]["n"],
        n_stops_per_vehicle=full["completed"],
        actual_end_s=timeline[-1]["time"] if timeline else 0,
        network_sha256=summary["network_sha256"], checkpoint_sha256=summary["checkpoint_sha256"],
        runtime_programs=summary["runtime_programs"],
        J0_green_min_s=15 if row["mode"].startswith("plain") else 5,
        J0_green_max_s=15 if row["mode"].startswith("plain") else 45,
        receiver_green_min_s=5 if row["mode"] == "v3" else 15,
        receiver_green_max_s=45 if row["mode"] == "v3" else 15,
        controller_class="PreannouncedWindow" if row["mode"] == "v3" else row["mode"],
        missing_metrics={}, quality_flags=[])
    for key, ids in sets.items():
        row[f"cohort_{key}_sha256"] = digest(sorted(ids))
    for key, count in (("waiting_E1_s", row["cohort_J2"]), ("full_waiting_s", row["cohort_full"])):
        if not count:
            row[key] = None
    for key, value in row.items():
        if value is None and (key.endswith("_pct") or key.endswith("_s") or key == "stops_per_vehicle"):
            row["missing_metrics"][key] = "empty_cohort_or_no_observed_samples"
    row["programmed_balance_residual"] = row["scheduled_vehicles"] - row["inserted_vehicles"] - row["pending_insertion"]
    row["inserted_balance_residual"] = row["inserted_vehicles"] - row["arrived"] - row["pending_active"]
    row["pending_balance_residual"] = row["pending"] - row["pending_active"] - row["pending_insertion"]
    if row["teleports"]:
        row["quality_flags"].append("teleports_observed")
    if any(row[k] for k in ("programmed_balance_residual", "inserted_balance_residual", "pending_balance_residual")):
        row["quality_flags"].append("vehicle_balance_residual_check_teleports_or_removals")
    if row["pending"]:
        row["quality_flags"].append("network_pending_at_end")
    if row["pending_J2"] or row["corridor_pending"]:
        row["quality_flags"].append("censored_corridor_cohort")
    if row["actual_end_s"] < END:
        row["quality_flags"].append("simulation_ended_before_horizon")
    coverage = physical_coverage(result)
    for tls, values in coverage.items():
        row[f"physical_window_count_{tls}"] = values["windows"]
        row[f"physical_arrival_window_{tls}_s"] = values["arrival_window_s"]
        row[f"physical_green_covered_{tls}_s"] = values["green_covered_s"]
    runs = {tls: [p for p in phases if p["tls"] == tls] for tls in ("J0", "J2", "J10", "J16")}
    validate_phases(runs)
    for tls, values in runs.items():
        if row["mode"].startswith("plain") or (row["mode"] == "only_ppo" and tls != "J0"):
            if any(p["duration"] != (4 if p["phase"] % 2 else 15) for p in values if not p["partial"]):
                raise ValueError(f"Fixed duration changed: {tls}")
    check_row(row)
    return row


def check_row(row):
    if row["mode"] not in MODES or row["constraint_violations"] != 0:
        raise ValueError("Unsupported mode or signal constraint violation")
    for suffix in ("J2", "full"):
        n, done, programmed = row[f"no_stop_{suffix}"], row[f"completed_{suffix}"], row[f"cohort_{suffix}"]
        if not 0 <= n <= done <= programmed:
            raise ValueError(f"Invalid cohort counts: {suffix}")
        for column, denominator in ((f"no_stop_{suffix}_pct", programmed),
                                    (f"no_stop_{suffix}_completed_pct", done)):
            expected = percentage(n, denominator)
            if row[column] != expected:
                raise ValueError(f"Incorrect percentage/denominator: {column}")
    if row["waiting_E1_all_observed_s"] is not None and row["waiting_E1_s"] is not None:
        if row["waiting_E1_all_observed_s"] < row["waiting_E1_s"]:
            raise ValueError("All-observed E1 waiting excludes historical waiting")
    if row["cohort_J2_observed_E1"] == 0 and row["waiting_E1_all_observed_s"] is not None:
        raise ValueError("Unobserved E1 cohort reported as observed waiting")


def worker(route, seed, mode, output):
    if WINDOW_CONTROLLERS["v3"] is not PreannouncedWindow:
        raise RuntimeError("V3 registry no longer selects the original controller")
    result = run(seed, mode, route_path=route) if mode == "v3" else baseline(seed, mode, route_path=route)
    payload = {"row": measured_row(result, route), "historical_row": historical_row(result),
               "summary": result[0],
               "canonical_traces": {name: canonical(rows) for name, rows in zip(TRACE_NAMES, result[1:])}}
    atomic_json(output, payload)


def execute_worker(route, seed, mode):
    with scratch("sumo_matched_worker_") as directory:
        output = Path(directory) / "result.json"
        command = [sys.executable, "-m", "tools.run_matched_experiment", "--worker-route", str(route),
                   "--worker-seed", str(seed), "--worker-mode", mode, "--worker-output", str(output)]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(f"Worker exited {process.returncode}:\n{(process.stdout + process.stderr)[-12000:]}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        payload["worker_command"] = command[:-1] + ["<temporary-result.json>"]
        return payload


def environment():
    import sumolib
    router = sumolib.checkBinary("duarouter")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch")}
    versions.update(python=sys.version.split()[0], sumo=subprocess.check_output([sumo_binary(), "--version"], text=True).splitlines()[0],
                    duarouter=subprocess.check_output([router, "--version"], text=True).splitlines()[0],
                    sumo_binary=sumo_binary(), duarouter_binary=router)
    return versions


def snapshot():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    paths = set(reference["protected"])
    paths.update(p.relative_to(ROOT).as_posix() for folder in ("sync", "policy", "tests")
                 for p in (ROOT / folder).glob("*.py"))
    paths.update(p.relative_to(ROOT).as_posix() for p in DEMAND.glob("*.rou.xml"))
    paths.update(("run_sync.py", "tools/compare_corridor.py", "tools/run_matched_experiment.py",
                  "tools/randomTrips.py", "analysis/corridor_full/reference.json",
                  "analysis/six_configurations/per_demand.csv", "analysis/six_configurations/validation.json"))
    for path, expected in reference["protected"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"Frozen protected resource changed: {path}")
    return {p: file_hash(ROOT / p) for p in sorted(paths)}


def manifest_core():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "master-slave-clean-v2":
        raise ValueError(f"Wrong branch: {branch}")
    return {"schema_version": 1, "branch": branch,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "protocol": {"levels": LEVELS, "modes": list(MODES), "horizon_s": END, "step_s": 1,
                         "SUMO_seed": 42, "offsets_sync": list(OFFSETS), "v3_delays_s": [12, 22, 32],
                         "green_bounds_s": [5, 45], "fixed_green_s": 15, "yellow_s": 4,
                         "deterministic": True, "training": False, "stop_speed_m_s": 0.1,
                         "cohort_max_depart_s": END - 180, "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
                         "generator": "tools/randomTrips.py", "generator_options": "historical defaults, --route-file, --validate",
                         "labels": LABELS, "reference_float_abs_tolerance": 1e-12},
            "environment": environment(), "resource_sha256": snapshot()}


def open_state(path, manifest):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS runs (experiment_id TEXT, level TEXT, seed INTEGER, mode TEXT, status TEXT, attempts INTEGER NOT NULL DEFAULT 0, row_json TEXT, evidence_json TEXT, error TEXT, started TEXT, finished TEXT, PRIMARY KEY(experiment_id,level,seed,mode))")
    identity = manifest["experiment_id"]
    existing = db.execute("SELECT value FROM metadata WHERE key='experiment_id'").fetchone()
    if existing and existing[0] != identity:
        db.close()
        raise ValueError("State belongs to a different protocol/code/resources")
    with db:
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('experiment_id',?)", (identity,))
    return db


def begin_attempt(db, identity, level, seed, mode):
    key = (identity, level, seed, mode)
    with db:
        db.execute("INSERT OR IGNORE INTO runs (experiment_id,level,seed,mode,status) VALUES (?,?,?,?,'pending')", key)
        state = db.execute("SELECT status,attempts FROM runs WHERE experiment_id=? AND level=? AND seed=? AND mode=?", key).fetchone()
        if state[0] == "done" or (state[0] == "failed" and state[1] >= 2):
            return False
        db.execute("UPDATE runs SET status='running',attempts=attempts+1,error=NULL,started=?,finished=NULL WHERE experiment_id=? AND level=? AND seed=? AND mode=?", (utc(), *key))
    return True


def finish_attempt(db, identity, level, seed, mode, row=None, evidence=None, error=None):
    with db:
        db.execute("UPDATE runs SET status=?,row_json=?,evidence_json=?,error=?,finished=? WHERE experiment_id=? AND level=? AND seed=? AND mode=?",
                   ("failed" if error else "done", json.dumps(row, ensure_ascii=False, allow_nan=False) if row else None,
                    json.dumps(evidence, ensure_ascii=False, allow_nan=False) if evidence else None,
                    error, utc(), identity, level, seed, mode))


def generator_command(seed, period, trips, route):
    return [sys.executable, str(GENERATOR), "-n", str(NET), "-b", "0", "-e", str(END),
            "-p", str(period), "-s", str(seed), "-o", str(trips), "-r", str(route), "--validate"]


def generate(seed, period, output):
    """Count raw trips before --validate replaces them; retain only final routed XML."""
    with scratch("sumo_matched_demand_") as directory:
        trips, route = Path(directory) / "demand.trips.xml", Path(directory) / "demand.rou.xml"
        command = generator_command(seed, period, trips, route)
        # The generator's --validate overwrites the trip file. Preserve the pre-router
        # count by instrumenting only subprocess.call in this isolated Python process.
        wrapper = '''import runpy, sys, subprocess, xml.etree.ElementTree as ET
original = subprocess.call
counts = []
def call(args, *a, **kw):
    if not counts:
        counts.append(sum(1 for _ in ET.parse(sys.argv[sys.argv.index("-o") + 1]).iter("trip")))
    code = original(args, *a, **kw)
    if code:
        raise RuntimeError("duarouter failed: " + str(code))
    return code
subprocess.call = call
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
print("MATCHED_TRIP_COUNT=" + str(counts[0]))
'''
        env = dict(os.environ, SUMO_HOME=str(SUMO_HOME))
        process = subprocess.run([sys.executable, "-c", wrapper, *command[1:]], cwd=ROOT, env=env,
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(f"Demand generation failed:\n{process.stdout}\n{process.stderr}")
        count = next(int(line.split("=", 1)[1]) for line in process.stdout.splitlines() if line.startswith("MATCHED_TRIP_COUNT="))
        programmed = sum(1 for _ in ET.parse(route).iter("vehicle"))
        if any(float(v.get("depart")) < 0 or float(v.get("depart")) >= END for v in ET.parse(route).iter("vehicle")):
            raise ValueError("Generated departures outside the horizon")
        if output.exists():
            raise FileExistsError(output)
        output.write_bytes(route.read_bytes())
        return {"attempted_trips": math.ceil(END / period), "generated_trips": count,
                "generation_losses": math.ceil(END / period) - count, "scheduled_vehicles": programmed,
                "routing_losses": count - programmed, "route_sha256": file_hash(output),
                "route_semantic_sha256": digest(route_content(output)),
                "generation_command": command, "router_seed": "duarouter default (unchanged)"}


def validate_references(output):
    evidence = {"demand_reproductions": [], "historical_runs": []}
    with scratch("sumo_matched_reference_") as directory:
        for level, period, reference in (("bajo", 2.0, ROOT / "maps/master_slave.rou.xml"),
                                          ("alto", 0.5, ROOT / "maps/master_slave_more_traffic.rou.xml")):
            route = Path(directory) / f"{level}.rou.xml"
            info = generate(42, period, route)
            if route_content(route) != route_content(reference):
                raise ValueError(f"Demand reference not reproduced: {level}")
            evidence["demand_reproductions"].append({"level": level, "seed": 42, "equal": True, **info})
    frozen = json.loads(REFERENCE.read_text(encoding="utf-8"))
    with SIX_CSV.open(encoding="utf-8", newline="") as handle:
        expected_rows = {r["mode"]: r for r in csv.DictReader(handle) if r["seed"] == "42" and r["mode"] in MODES}
    for mode in MODES:
        print(f"VALIDATE historical seed=42 mode={mode}", flush=True)
        payload = execute_worker(DEMAND / "seed_42.rou.xml", 42, mode)
        actual = payload["historical_row"]
        # Historical order is 6 for V3 in the six-configuration CSV.
        actual["order"] = int(expected_rows[mode]["order"])
        for key, serialized in expected_rows[mode].items():
            value = actual[key]
            if serialized == "":
                expected = None
            else:
                try:
                    expected = json.loads(serialized)
                except json.JSONDecodeError:
                    expected = {"True": True, "False": False}.get(serialized, serialized)
            compare(value, expected, f"historical.{mode}.{key}")
        sources = ["analysis/six_configurations/per_demand.csv"]
        if mode == "v3":
            compare(payload["summary"], frozen["runs"]["seed42_v3_full"]["summary"])
            compare(payload["canonical_traces"], frozen["runs"]["seed42_v3_full"]["traces"])
            sources.append("analysis/corridor_full/reference.json")
        additional = ROOT / f"results/comparison_six_20261003/seed42_{mode}.json"
        if additional.is_file():
            saved = json.loads(additional.read_text(encoding="utf-8"))
            compare(payload["summary"], saved["summary"])
            compare(payload["canonical_traces"], saved["canonical_traces"])
            sources.append(additional.relative_to(ROOT).as_posix())
        evidence["historical_runs"].append({"mode": mode, "equal": True, "sources": sources,
                                            "canonical_traces": payload["canonical_traces"]})
        atomic_json(output / "validation.json", {"status": "preflight_in_progress", **evidence})
    return evidence


def prepare_demands(output):
    path = output / "demand_manifest.csv"
    existing = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = {r["demand_id"]: r for r in csv.DictReader(handle)}
    rows = []
    (output / "demand").mkdir(exist_ok=True)
    for level, settings in LEVELS.items():
        for seed in settings["seeds"]:
            demand_id = f"{level}_{seed}"
            relative = f"demand/{demand_id}.rou.xml"
            route = output / relative
            if demand_id in existing:
                row = existing[demand_id]
                if not route.is_file() or file_hash(route) != row["route_sha256"]:
                    raise ValueError(f"Frozen demand changed or missing: {demand_id}")
            else:
                print(f"GENERATE {demand_id} period={settings['period_s']}", flush=True)
                # A crash between route creation and manifest update leaves an orphan;
                # regenerate in temp and verify semantic equality before adopting it.
                with scratch("sumo_matched_publish_") as directory:
                    candidate = Path(directory) / "route.rou.xml"
                    info = generate(seed, settings["period_s"], candidate)
                    if route.exists():
                        if route_content(route) != route_content(candidate):
                            raise ValueError(f"Orphan demand differs: {demand_id}")
                        info["route_sha256"] = file_hash(route)
                    else:
                        temporary = route.with_suffix(".tmp")
                        temporary.write_bytes(candidate.read_bytes())
                        os.replace(temporary, route)
                row = {"demand_id": demand_id, "traffic_level": level, "seed": seed, "route_path": relative,
                       "generation_period_s": settings["period_s"],
                       "nominal_attempts_per_hour": 3600 / settings["period_s"], **info}
            rows.append(row)
            # Preserve later existing demands during a partially resumed traversal.
            all_rows = {**existing, **{r["demand_id"]: r for r in rows}}
            atomic_csv(path, list(all_rows.values()))
    return rows


def export(output, db, manifest, preflight):
    entries = db.execute("SELECT level,seed,mode,row_json,status,error,attempts FROM runs WHERE experiment_id=?", (manifest["experiment_id"],)).fetchall()
    rows = [json.loads(e[3]) for e in entries if e[4] == "done"]
    rows.sort(key=lambda r: (list(LEVELS).index(r["traffic_level"]), r["seed"], MODES.index(r["mode"])))
    groups = {}
    for row in rows:
        check_row(row)
        key = (row["traffic_level"], row["seed"])
        signature = [row[k] for k in ("route_sha256", "scheduled_vehicles", "cohort_J2_sha256", "cohort_full_sha256", "cohort_strict_sha256")]
        if key in groups and groups[key] != signature:
            raise ValueError(f"Matched demands/cohorts differ: {key}")
        groups[key] = signature
    if snapshot() != manifest["resource_sha256"]:
        raise ValueError("Protected resources/code changed during the experiment")
    atomic_csv(output / "per_demand.csv", rows)
    complete = len(rows) == 240 and len(groups) == 60
    validation = {"experiment_id": manifest["experiment_id"], "status": "complete" if complete else "incomplete",
                  "completed_runs": len(rows), "expected_runs": 240, "matched_demands_and_programmed_cohorts": True,
                  "protected_hashes_unchanged": True, "preflight": preflight,
                  "flagged_runs": [{"run_id": r["run_id"], "flags": r["quality_flags"]} for r in rows if r["quality_flags"]],
                  "unfinished_runs": [{"level": e[0], "seed": e[1], "mode": e[2], "status": e[4], "error": e[5], "attempts": e[6]} for e in entries if e[4] != "done"]}
    atomic_json(output / "validation.json", validation)
    return complete


def execute_batch(output, db, manifest, demands, preflight):
    identity = manifest["experiment_id"]
    for demand in demands:
        level, seed = demand["traffic_level"], int(demand["seed"])
        route = output / demand["route_path"]
        for mode in MODES:
            while begin_attempt(db, identity, level, seed, mode):
                print(f"START {level} seed={seed} mode={mode} {utc()}", flush=True)
                started = time.monotonic()
                try:
                    if snapshot() != manifest["resource_sha256"] or file_hash(route) != demand["route_sha256"]:
                        raise ValueError("Resources or demand changed before simulation")
                    payload = execute_worker(route, seed, mode)
                    row = payload.pop("row")
                    if row["route_sha256"] != demand["route_sha256"]:
                        raise ValueError("Worker used a different demand")
                    for key in ("generation_period_s", "nominal_attempts_per_hour", "attempted_trips", "generated_trips", "generation_losses", "routing_losses"):
                        row[key] = float(demand[key]) if key in ("generation_period_s", "nominal_attempts_per_hour") else int(demand[key])
                    row.update(experiment_id=identity, run_id=f"{identity}:{level}:{seed}:{mode}",
                               traffic_level=level, demand_id=demand["demand_id"], route_path=demand["route_path"],
                               route_semantic_sha256=demand["route_semantic_sha256"],
                               code_sha256=digest(manifest["resource_sha256"]), environment=manifest["environment"],
                               run_command=[sys.executable, "-m", "tools.run_matched_experiment", "--stage", "run", "--output-dir", str(output)],
                               sumo_command_template=[sumo_binary(), "--net-file", "<temporary network with recorded offsets>",
                                                      "--route-files", str(route), "--begin", "0", "--end", "3600", "--step-length", "1", "--seed", "42",
                                                      "--no-step-log", "true", "--no-warnings", "true"],
                               elapsed_wall_s=round(time.monotonic() - started, 3))
                    finish_attempt(db, identity, level, seed, mode, row, payload)
                    print(f"DONE {level} seed={seed} mode={mode} waiting={row['network_waiting_s']} pending={row['pending']} elapsed={row['elapsed_wall_s']}s", flush=True)
                except Exception:
                    error = traceback.format_exc()
                    finish_attempt(db, identity, level, seed, mode, error=error)
                    print(f"FAILED {level} seed={seed} mode={mode}\n{error}", flush=True)
                export(output, db, manifest, preflight)
    return export(output, db, manifest, preflight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "validate", "prepare", "run", "export"), default="inspect")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-route", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_route is not None:
        worker(args.worker_route, args.worker_seed, args.worker_mode, args.worker_output)
        return
    core = manifest_core()
    identity = digest(core)
    if args.stage == "inspect":
        print(json.dumps({"experiment_id": identity, **core}, ensure_ascii=False, indent=2))
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    manifest = {**core, "experiment_id": identity, "created_UTC": utc()}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["experiment_id"] != identity or {k: saved[k] for k in core} != core:
            raise ValueError("Existing experiment has different code/protocol/resources; use a new directory")
        manifest = saved
    else:
        atomic_json(path, manifest)
    db = open_state(output / "runs.sqlite", manifest)
    try:
        entry = db.execute("SELECT value FROM metadata WHERE key='preflight'").fetchone()
        if args.stage == "export":
            complete = export(output, db, manifest, json.loads(entry[0]) if entry else None)
        else:
            if entry:
                preflight = json.loads(entry[0])
            else:
                preflight = validate_references(output)
                with db:
                    db.execute("INSERT INTO metadata VALUES ('preflight',?)", (json.dumps(preflight),))
            if args.stage == "validate":
                atomic_json(output / "validation.json", {"status": "preflight_passed", "preflight": preflight})
                print("PASS: two demand references and four historical simulations", flush=True)
                return
            demands = prepare_demands(output)
            if args.stage == "prepare":
                print("READY: 60 frozen demands; no new batch simulations executed", flush=True)
                return
            complete = execute_batch(output, db, manifest, demands, preflight)
        print("COMPLETE: 240 rows" if complete else "INCOMPLETE: inspect validation.json and resume with --stage run", flush=True)
        if not complete:
            raise SystemExit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
