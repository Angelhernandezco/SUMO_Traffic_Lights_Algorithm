"""Run deterministic PPO J0 and synchronize the complete northbound corridor."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sync.common import ROOT
from sync.metrics import write_csv
from sync.simulation import run
from sync.v3_variants import WINDOW_CONTROLLERS

TRACE_NAMES = ("releases", "vehicles", "events", "timeline", "actions", "phases", "per_vehicle")


def save_result(result, output, traces=False):
    summary, *rows = result
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"seed{summary['seed']}_{summary['mode']}_full"
    path = output / f"{stem}.json"
    if path.exists():
        raise FileExistsError(f"Result already exists: {path}")
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if traces:
        for suffix, values in zip(TRACE_NAMES, rows):
            write_csv(output / f"{stem}_{suffix}.csv", values)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("v1", "v2", *WINDOW_CONTROLLERS), required=True)
    parser.add_argument("--seed", type=int, choices=range(42, 47), default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--delay", type=int, default=100, help="GUI delay in milliseconds")
    parser.add_argument("--travel-times", nargs=3, type=int, default=(11, 10, 10),
                        help="V2 only: travel seconds J0->J2, J2->J10, J10->J16")
    parser.add_argument("--traces", action="store_true", help="Export detailed CSV traces")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.delay < 0 or any(t <= 0 for t in args.travel_times):
        parser.error("Delay must be nonnegative and travel times positive")
    if args.mode != "v2" and tuple(args.travel_times) != (11, 10, 10):
        parser.error("--travel-times is configurable only for V2")
    output = args.output_dir or ROOT / "results" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
    expected = output / f"seed{args.seed}_{args.mode}_full.json"
    if expected.exists():
        parser.error(f"Result already exists: {expected}")
    result = run(args.seed, args.mode, args.travel_times, args.gui, args.delay)
    path = save_result(result, output, args.traces)
    print(json.dumps(result[0], indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
