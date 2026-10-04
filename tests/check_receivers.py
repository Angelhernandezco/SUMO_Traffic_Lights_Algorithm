"""Native SUMO checks for receiver phase rotation, causal extension and legal limits."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sync.common import traci, RECEIVERS, OFFSETS, static_net, sumo_binary
from sync.v1 import J2Advance
from sync.v3 import PreannouncedWindow


def check(tls, mode, empty=False):
    phase = RECEIVERS[tls][0]
    delay = {"J2": 12, "J10": 22, "J16": 32}[tls]
    with tempfile.TemporaryDirectory(prefix="sumo_full_check_") as tmp:
        net = Path(tmp) / "wave.net.xml"
        static_net(net, OFFSETS)
        traci.start([sumo_binary(), "--net-file", str(net), "--step-length", "1", "--seed", "42",
                     "--no-step-log", "true", "--no-warnings", "true"])
        try:
            audit = SimpleNamespace(crossings=[], last_release_id=0)
            ctrl = (J2Advance(delay, tls, phase) if mode == "v1" else
                    PreannouncedWindow(audit, tls, phase, delay))
            observed = {}
            for now in range(141):
                if now:
                    traci.simulationStep()
                    if mode == "v3" and not empty and now in (45, 54, 75):
                        audit.crossings.append({"release_id": 0, "time": now,
                                               "vehicle_id": str(now), "basis": "synthetic"})
                    ctrl.tick()
                    observed[now] = traci.trafficlight.getPhase(tls)
                if now == 35 and mode == "v3":
                    ctrl.announce(36, 5)
                    assert ctrl.events[0]["provisional_front"] == 45 + delay
                if now == 45:
                    if mode == "v1":
                        ctrl.release(45)
                    else:
                        ctrl.prepare(45)
                if now == 80 and mode == "v3":
                    ctrl.release_finished(0)
            ctrl.finish()
            assert len(ctrl.events) == 1
            # The initial offset may start halfway through a phase.
            for i, run in enumerate(ctrl.phase_runs):
                if i == 0 and tls != "J2":
                    continue
                assert run["duration"] == 4 if run["phase"] % 2 else 5 <= run["duration"] <= 45
            assert all(b["phase"] == (a["phase"] + 1) % 8
                       for a, b in zip(ctrl.phase_runs, ctrl.phase_runs[1:]))
            event = ctrl.events[0]
            if mode == "v3":
                assert event["closed"] and event["first_cross_updates"] == (not empty)
                if empty:
                    assert event["status"] == "empty_release"
                else:
                    assert event["front_target"] == 45 + delay and event["tail_target"] == 75 + delay
                    if event.get("front_reachable_at_first_cross"):
                        assert observed[event["front_target"]] == phase
            return {"tls": tls, "mode": mode, "empty": empty,
                    "event": event, "phase_runs": ctrl.phase_runs, "violations": 0}
        finally:
            traci.close()


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    rows = [check(t, m, e) for t in RECEIVERS for m, e in
            (("v1", False), ("v3", False), ("v3", True))]
    Path("results/mechanism_receivers.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("PASS: 9 native SUMO receiver/cause/empty/order/duration checks", flush=True)
