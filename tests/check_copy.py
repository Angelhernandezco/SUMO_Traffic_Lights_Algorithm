"""Native SUMO checks for exact green copying and legal timing, without PPO."""

import json
from pathlib import Path
import tempfile

from sync.common import traci, RECEIVERS, static_net, sumo_binary
from sync.v2 import CopyGreen


def main():
    cases = [
        ("minimum_5", [(1, 5)], [11, 21, 31]),
        ("example_25", [(1, 25)], [11, 21, 31]),
        ("maximum_45", [(1, 45)], [11, 21, 31]),
        ("unreachable_front", [(1, 25)], [4, 8, 12]),
        ("current_green_25", [(22, 25)], [11, 21, 31]),
        ("elapsed_green_5", [(25, 5)], [11, 21, 31]),
        ("two_copies_without_merging", [(1, 45), (20, 5)], [11, 21, 31]),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="sumo_copy_checks_") as tmp:
        net = Path(tmp) / "wave.net.xml"
        static_net(net, (0, 0, 0, 0))
        for name, releases, delays in cases:
            traci.start([sumo_binary(),
                                "--net-file", str(net), "--step-length", "1",
                                "--no-step-log", "true", "--no-warnings", "true"])
            try:
                controllers = {tls: CopyGreen(tls, delay)
                               for tls, delay in zip(RECEIVERS, delays)}
                for now in range(1, 251):
                    traci.simulationStep()
                    for controller in controllers.values():
                        controller.tick()
                    for at, green in releases:
                        if at == now:
                            for controller in controllers.values():
                                controller.release(at, green, at - 1)
                for controller in controllers.values():
                    controller.finish()
                    assert len(controller.events) == len(releases)
                    assert all(e["status"] == "copied" for e in controller.events)
                    assert all(e["receiver_green_executed"] == e["J0_green"] for e in controller.events)
                    assert all(r["duration"] == 4 if r["phase"] % 2 else 5 <= r["duration"] <= 45
                               for r in controller.phase_runs if not r["partial"])
                    assert all(b["phase"] == (a["phase"] + 1) % 8
                               for a, b in zip(controller.phase_runs, controller.phase_runs[1:]))
                if name == "unreachable_front":
                    assert not controllers["J2"].events[0]["start_target_reachable"]
                    assert controllers["J2"].events[0]["target_error"] > 0
                if name == "current_green_25":
                    assert controllers["J2"].events[0]["using_current_green"]
                if name == "elapsed_green_5":
                    assert not controllers["J2"].events[0]["using_current_green"]
                results.append({"case": name, "passed": True, "events": {
                    tls: [{"green": e["J0_green"], "copied": e["receiver_green_executed"],
                           "target": e["target"], "opening": e["actual_start"]}
                          for e in c.events] for tls, c in controllers.items()}})
            finally:
                traci.close()
    output = Path("results/mechanism_copy.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    main()
