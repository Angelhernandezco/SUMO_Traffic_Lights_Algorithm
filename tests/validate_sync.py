"""Replay full runs against the frozen summaries and canonical CSV signatures."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from run_sync import TRACE_NAMES, save_result
from sync.common import ROOT
from sync.simulation import run

REFERENCE = ROOT / "analysis/corridor_full/reference.json"


def canonical(rows):
    """Match CSV serialization, ignoring row order and dictionary key order."""
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    buffer.seek(0)
    values = sorted(json.dumps(r, sort_keys=True, separators=(",", ":"))
                    for r in csv.DictReader(buffer))
    return {"rows": len(rows), "sha256": hashlib.sha256("\n".join(values).encode()).hexdigest()}


def compare(actual, expected, path="summary"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (path, actual.keys() ^ expected.keys())
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def protected(reference):
    for relative, expected in reference["protected"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 47)))
    parser.add_argument("--modes", nargs="+", default=["v1", "v2", "v3"])
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--traces", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/validation")
    args = parser.parse_args()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    protected(reference)
    validations = []
    for seed in args.seeds:
        for mode in args.modes:
            key = f"seed{seed}_{mode}_full"
            result = run(seed, mode, gui=args.gui, delay=0)
            save_result(result, args.output_dir, args.traces)
            expected = reference["runs"][key]
            compare(result[0], expected["summary"])
            signatures = {name: canonical(rows) for name, rows in zip(TRACE_NAMES, result[1:])}
            compare(signatures, expected["traces"], "traces")
            validations.append({"run": key, "summary_equal": True, "all_seven_traces_equal": True,
                                "constraint_violations": result[0]["constraint_violations"]})
            (args.output_dir / "validation.json").write_text(json.dumps(validations, indent=2), encoding="utf-8")
            print(f"PASS {key}: summary + seven canonical traces", flush=True)
    protected(reference)
    print(f"PASS {len(validations)} runs; protected resources unchanged", flush=True)


if __name__ == "__main__":
    main()
