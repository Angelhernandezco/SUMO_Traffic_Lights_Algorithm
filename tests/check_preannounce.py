"""Real SUMO phase/timestamp checks with synthetic, causal crossing notices."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sync.common import traci, OFFSETS, static_net, sumo_binary
from sync.v3 import PreannouncedWindow
import sync.v3 as v3


def check(name, notice_at, previous_green, crosses, finished_at, stop=120):
    with tempfile.TemporaryDirectory(prefix="sumo_v3_check_") as tmp:
        net = Path(tmp) / "test.net.xml"
        static_net(net, OFFSETS)
        traci.start([sumo_binary(), "--net-file", str(net), "--begin", "0",
                     "--step-length", "1", "--seed", "42", "--no-step-log", "true"])
        try:
            audit = SimpleNamespace(crossings=[], last_release_id=0)
            control = PreannouncedWindow(audit)
            phases = {}
            for now in range(stop + 1):
                if now:
                    traci.simulationStep()
                    if now in crosses:
                        audit.crossings.append({"release_id": 0, "time": now,
                                                "vehicle_id": str(now), "basis": "synthetic"})
                    control.tick()
                    phases[now] = traci.trafficlight.getPhase("J2")
                if now == notice_at:
                    control.announce(now + 1, previous_green)
                if now == finished_at:
                    control.release_finished(0)
            control.finish()
            event = control.events[0]
            for run in control.phase_runs:
                assert (run["duration"] == 4 if run["phase"] % 2 else
                        5 <= run["duration"] <= 45), run
            assert all(b["phase"] == (a["phase"] + 1) % 8
                       for a, b in zip(control.phase_runs, control.phase_runs[1:]))
            assert len(control.events) == 1 and event["closed"]
            assert event["first_cross_updates"] == bool(crosses)
            if crosses and event.get("front_reachable_at_first_cross"):
                assert phases[event["front_target"]] == 2
            return {"case": name, "event": event,
                    "corridor_runs": [r for r in control.phase_runs if r["phase"] == 2],
                    "front_green": phases.get(event["front_target"]) == 2,
                    "tail_green": phases.get(event["tail_target"]) == 2,
                    "constraint_violations": 0, "duplicate_events": 0}
        finally:
            traci.close()


def main():
    rows = [check("natural_green_no_cut", 0, 5, [10, 11], 18),
            check("late_front_earliest_legal_opening", 35, 5, [45], 58),
            check("front_correction_cancels_unused_cut", 35, 5, [55], 59),
            check("empty_preparation_closes", 35, 5, [], 58),
            check("causal_tail_and_45s_cap", 21, 9, [35, 44, 54], 64)]
    v3.END = 100
    try:
        terminal = check("terminal_request_stays_closed", 85, 5, [95], 100)
        assert terminal["event"]["status"] == "censored_window"
        assert not terminal["event"]["green_cuts"]
        rows.append(terminal)
    finally:
        v3.END = 3600
    by_name = {row["case"]: row for row in rows}
    assert not by_name["natural_green_no_cut"]["event"]["green_cuts"]
    late = by_name["late_front_earliest_legal_opening"]["event"]
    assert not late["front_reachable_at_announcement"]
    assert late["actual_start"] == late["earliest_start_at_announcement"] == 66
    corrected = by_name["front_correction_cancels_unused_cut"]
    assert corrected["event"]["actual_start"] == 67 and corrected["front_green"]
    assert any(c.get("cancelled") for c in corrected["event"]["green_cuts"])
    empty = by_name["empty_preparation_closes"]["event"]
    assert empty["status"] == "empty_release" and empty["closed_at"] == 58
    cap = by_name["causal_tail_and_45s_cap"]
    assert cap["front_green"] and not cap["tail_green"]
    assert any(r["duration"] == 45 for r in cap["corridor_runs"])
    assert cap["event"]["tail_beyond_45"]
    output = Path("results/mechanism_preannounce.json")
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("PASS: six SUMO mechanism checks, no phase/duration/order violations")


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    main()
