"""Targeted checks of cut allocation; real phase legality is checked per run."""
from unittest.mock import patch
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from sync.common import traci, static_net, sumo_binary
from sync.v1 import J2Advance
from sync.v3_variants import (QueueBalancedWindow, TailClosedWindow,
                              DeferredQueueWindow, FullQueueBalancedWindow, AlignedQueueWindow,
                              ArrivalGuardWindow, DoubleSampleGuardWindow)


def native_tail_check():
    with tempfile.TemporaryDirectory(prefix="sumo_v3_tail_") as tmp:
        net = Path(tmp) / "test.net.xml"
        static_net(net)
        traci.start([sumo_binary(), "--net-file", str(net), "--step-length", "1",
                     "--seed", "42", "--no-step-log", "true", "--no-warnings", "true"])
        try:
            audit = SimpleNamespace(crossings=[], last_release_id=0)
            ctrl = TailClosedWindow(audit, "J2", 2, 12)
            ctrl.announce(1, 5)
            observed = {}
            for now in range(1, 91):
                traci.simulationStep()
                if now in (10, 11):
                    audit.crossings.append({"release_id": 0, "time": now, "vehicle_id": str(now)})
                ctrl.tick()
                if now == 18:
                    ctrl.release_finished(0)
                observed[now] = traci.trafficlight.getPhase("J2")
            ctrl.finish()
            event = ctrl.events[0]
            assert observed[event["front_target"]] == observed[event["tail_target"]] == 2
            assert "corridor_early_close_at" in event and event["unused_corridor_seconds_removed"] > 0
            assert all(r["duration"] == 4 if r["phase"] % 2 else 5 <= r["duration"] <= 45
                       for r in ctrl.phase_runs)
            assert all(b["phase"] == (a["phase"] + 1) % 8
                       for a, b in zip(ctrl.phase_runs, ctrl.phase_runs[1:]))
            print("PASS: native causal early close, front/tail covered, minimum green/full yellow/cyclic order")
        finally:
            traci.close()


def native_deferred_check():
    with tempfile.TemporaryDirectory(prefix="sumo_v3_deferred_") as tmp:
        net = Path(tmp) / "test.net.xml"
        static_net(net)
        traci.start([sumo_binary(), "--net-file", str(net), "--step-length", "1",
                     "--seed", "42", "--no-step-log", "true", "--no-warnings", "true"])
        try:
            audit = SimpleNamespace(crossings=[], last_release_id=0)
            ctrl = DeferredQueueWindow(audit, "J2", 2, 12)
            phases = {}
            for now in range(1, 121):
                traci.simulationStep()
                if now == 60:
                    audit.crossings.append({"release_id": 0, "time": now, "vehicle_id": "front"})
                ctrl.tick()
                if now == 35:
                    ctrl.announce(36, 20)
                    assert ctrl.events[0]["front_reachable_at_announcement"]
                    assert not ctrl.events[0]["green_cuts"]
                if now == 68:
                    ctrl.release_finished(0)
                phases[now] = traci.trafficlight.getPhase("J2")
            ctrl.finish()
            event = ctrl.events[0]
            assert 35 < event["cuts_committed_at"] < 60
            assert phases[event["front_target"]] == 2
            assert len(ctrl.events) == 1 and event["closed"] and not ctrl.deferred
            assert all(r["duration"] == 4 if r["phase"] % 2 else 5 <= r["duration"] <= 45
                       for r in ctrl.phase_runs)
            assert all(b["phase"] == (a["phase"] + 1) % 8
                       for a, b in zip(ctrl.phase_runs, ctrl.phase_runs[1:]))
            print("PASS: native deferred cuts, legal reachable front, one causal event, full yellow/cyclic order")
        finally:
            traci.close()


def native_full_queue_check():
    with tempfile.TemporaryDirectory(prefix="sumo_v3_fullqueue_") as tmp:
        net = Path(tmp) / "test.net.xml"
        static_net(net)
        traci.start([sumo_binary(), "--net-file", str(net), "--step-length", "1",
                     "--seed", "42", "--no-step-log", "true", "--no-warnings", "true"])
        try:
            audit = SimpleNamespace(crossings=[], last_release_id=0)
            ctrl = FullQueueBalancedWindow(audit, "J10", 4, 22)
            phases = {}
            for now in range(1, 121):
                traci.simulationStep()
                if now == 45:
                    audit.crossings.append({"release_id": 0, "time": now, "vehicle_id": "front"})
                ctrl.tick()
                if now == 35:
                    ctrl.announce(36, 5)
                if now == 54:
                    ctrl.release_finished(0)
                phases[now] = traci.trafficlight.getPhase("J10")
            ctrl.finish()
            event = ctrl.events[0]
            assert set(ctrl.phase_lanes) == {0, 2, 6} and all(ctrl.phase_lanes.values())
            assert phases[event["front_target"]] == 4
            assert len(ctrl.events) == 1 and event["closed"]
            assert all(r["duration"] == 4 if r["phase"] % 2 else 5 <= r["duration"] <= 45
                       for r in ctrl.phase_runs[1:])  # Initial offset is censored.
            assert all(b["phase"] == (a["phase"] + 1) % 8
                       for a, b in zip(ctrl.phase_runs, ctrl.phase_runs[1:]))
            print("PASS: native full queue allocation, rotated J10 phase, causal front, legal order/durations")
        finally:
            traci.close()


def native_aligned_check():
    with tempfile.TemporaryDirectory(prefix="sumo_v3_aligned_") as tmp:
        net = Path(tmp) / "test.net.xml"
        static_net(net)
        traci.start([sumo_binary(), "--net-file", str(net), "--step-length", "1",
                     "--seed", "42", "--no-step-log", "true", "--no-warnings", "true"])
        try:
            audit = SimpleNamespace(crossings=[], last_release_id=0)
            ctrl = AlignedQueueWindow(audit, "J2", 2, 12)
            ctrl.announce(1, 45)
            phases = {}
            for now in range(1, 151):
                traci.simulationStep()
                if now in (50, 51):
                    audit.crossings.append({"release_id": 0, "time": now, "vehicle_id": str(now)})
                ctrl.tick()
                if now == 59:
                    ctrl.release_finished(0)
                phases[now] = traci.trafficlight.getPhase("J2")
            ctrl.finish()
            event = ctrl.events[0]
            assert event["requested_delay"] > event["planned_delay"] > 0
            assert event["actual_start"] > event["baseline_next_start"]
            assert phases[event["front_target"]] == phases[event["tail_target"]] == 2
            assert any(c.get("seconds_added", 0) for c in event["green_cuts"])
            assert any(r["duration"] == 45 and r["phase"] != 2 for r in ctrl.phase_runs)
            assert all(r["duration"] == 4 if r["phase"] % 2 else 5 <= r["duration"] <= 45
                       for r in ctrl.phase_runs)
            assert all(b["phase"] == (a["phase"] + 1) % 8
                       for a, b in zip(ctrl.phase_runs, ctrl.phase_runs[1:]))
            print("PASS: native early-opening delay, 45s bound, front/tail covered, full yellow/cyclic order")
        finally:
            traci.close()


def native_guard_check(controller_class=ArrivalGuardWindow):
    with tempfile.TemporaryDirectory(prefix="sumo_v3_guard_") as tmp:
        net = Path(tmp) / "test.net.xml"
        static_net(net)
        traci.start([sumo_binary(), "--net-file", str(net), "--step-length", "1",
                     "--seed", "42", "--no-step-log", "true", "--no-warnings", "true"])
        try:
            audit = SimpleNamespace(crossings=[], last_release_id=0)
            ctrl = controller_class(audit, "J2", 2, 12)
            phases = {}
            for now in range(1, 121):
                traci.simulationStep()
                if now == 56:
                    audit.crossings.append({"release_id": 0, "time": now, "vehicle_id": "front"})
                ctrl.tick()
                if now == 35:
                    ctrl.announce(36, 16)
                if now == 64:
                    ctrl.release_finished(0)
                phases[now] = traci.trafficlight.getPhase("J2")
            ctrl.finish()
            event = ctrl.events[0]
            assert event["target"] == event["front_target"] == 68
            assert event["actual_start"] == 68-ctrl.opening_guard_s
            assert phases[68-ctrl.opening_guard_s] == phases[68] == 2
            assert event["opening_guard_s"] == ctrl.opening_guard_s and event["first_cross_updates"] == 1
            assert all(r["duration"] == 4 if r["phase"] % 2 else 5 <= r["duration"] <= 45
                       for r in ctrl.phase_runs)
            assert all(b["phase"] == (a["phase"] + 1) % 8
                       for a, b in zip(ctrl.phase_runs, ctrl.phase_runs[1:]))
            print(f"PASS: native {ctrl.opening_guard_s}s opening guard, unchanged +12 target, causal correction, legal order/durations")
        finally:
            traci.close()


def main():
    ctrl = object.__new__(QueueBalancedWindow)
    ctrl.tls = "J2"
    ctrl.phase_lanes = {0: ["busy"], 4: ["empty"], 6: ["mixed"]}
    path = [(0, 10, 5, True), (1, 4, 0, False), (2, 15, 0, False),
            (3, 4, 0, False), (4, 15, 10, False), (5, 4, 0, False),
            (6, 15, 10, False), (7, 4, 0, False)]
    queues = {"busy": 8, "empty": 0, "mixed": 3}
    with patch.object(J2Advance, "_remaining_path", return_value=(path, 90)), \
         patch("sync.v3_variants.traci.lane.getLastStepHaltingNumber", side_effect=queues.__getitem__), \
         patch("sync.v3_variants.traci.lane.getLastStepVehicleNumber", side_effect=queues.__getitem__):
        ranked, baseline = ctrl._remaining_path()
        assert [p[0] for p in ranked[:3]] == [4, 6, 0]
        assert baseline == 90 and sorted(ranked) == sorted(path)
        assert sum(p[2] for p in ranked) == sum(p[2] for p in path)
        ctrl.tls = "J10"
        assert ctrl._remaining_path() == (path, 90)
    print("PASS: queue ranking, same path/authority, unchanged J10/J16 allocation")
    native_tail_check()
    native_deferred_check()
    native_full_queue_check()
    native_aligned_check()
    native_guard_check()
    native_guard_check(DoubleSampleGuardWindow)
    output = Path("results/v3_balance/native_checks.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": ["queue_allocation", "causal_tail_close", "deferred_preparation",
                                            "rotated_J10_queue_allocation", "early_opening_alignment",
                                            "one_second_guard", "two_second_guard"],
                                  "unit_cases": 1, "native_SUMO_cases": 6}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
