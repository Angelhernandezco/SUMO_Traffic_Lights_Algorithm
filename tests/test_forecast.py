"""Behavioral checks for the read-only corridor observer."""

import math
import os
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from policy.forecast import (
    ForecastStore,
    Platoon,
    ShadowForecast,
    TrackedVehicle,
    baseline_eta,
    classify_route,
)


ROOT = Path(__file__).resolve().parents[1]


class ForecastUnitTests(unittest.TestCase):
    def test_terminal_edges_do_not_create_slave_arrivals(self):
        cases = (
            (("-E0", "E1"), "e1_only", (False, False, False)),
            (("-E0", "E1", "E5"), "e1_e5", (True, False, False)),
            (("-E0", "E1", "E5", "E10"), "e1_e5_e10", (True, True, False)),
            (("-E0", "E1", "E5", "E10", "E13"), "e1_e5_e10", (True, True, True)),
            (("E2", "E1", "E7"), "e1_only", (True, False, False)),
        )
        for route, expected_class, expected_applicable in cases:
            with self.subTest(route=route):
                route_class, applicable = classify_route(route, 1)
                self.assertEqual(route_class, expected_class)
                self.assertEqual(tuple(applicable.values()), expected_applicable)

    def test_eta_uses_remaining_distance_and_signed_error(self):
        predicted = baseline_eta(100.0, 103.06 - 20.0, 16.67)
        self.assertAlmostEqual(predicted, 104.9826, places=3)
        self.assertGreater(predicted - 104.0, 0)  # prediction late
        self.assertLess(predicted - 110.0, 0)  # prediction early
        self.assertIsNone(baseline_eta(100.0, -1.0, 16.67))
        self.assertIsNone(baseline_eta(100.0, 10.0, 0.0))
        self.assertIsNone(baseline_eta(100.0, math.inf, 16.67))

    def test_gap_and_total_window_are_both_bounded(self):
        platoon = Platoon(1, 10.0, 12.0, ["a", "b"])
        self.assertTrue(platoon.accepts(15.0))
        self.assertFalse(platoon.accepts(15.01))
        platoon.last_time = 24.0
        self.assertTrue(platoon.accepts(25.0))
        self.assertFalse(platoon.accepts(25.01))

    def test_store_aggregates_only_applicable_members(self):
        store = ForecastStore()
        platoon = Platoon(1, 10.0, 12.0, ["a", "b", "c"])
        def vehicle(vid, eta, status):
            return TrackedVehicle(vid, 10.0, "-E0", ("-E0", "E1", "E5"),
                                  "e1_e5", 1, 2.0,
                                  {"J2": eta, "J10": None, "J16": None},
                                  {"J2": 80.0, "J10": None, "J16": None},
                                  {"J2": status, "J10": "no_aplicable", "J16": "no_aplicable"})
        vehicles = {"a": vehicle("a", 17.0, "pendiente"),
                    "b": vehicle("b", 19.0, "pendiente"),
                    "c": vehicle("c", None, "no_aplicable")}
        store.update(platoon, vehicles, 12.0)
        forecast = store.by_slave["J2"][1]
        self.assertEqual(forecast.expected_vehicle_count, 2)
        self.assertEqual(forecast.predicted_arrival, 18.0)
        self.assertEqual((forecast.predicted_window_start, forecast.predicted_window_end), (17.0, 19.0))
        self.assertEqual(forecast.dispersion, 1.0)
        self.assertEqual(forecast.quality, "completa")
        vehicles["b"].eta_by_slave["J2"] = None
        store.update(platoon, vehicles, 13.0)
        self.assertEqual(store.by_slave["J2"][1].quality, "parcial")
        self.assertEqual(store.by_slave["J2"][1].revision, 2)


class ForecastTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.observer = ShadowForecast()
        self.traci = mock.Mock()
        self.traci.vehicle.getIDList.return_value = ()
        self.traci.simulation.getDepartedIDList.return_value = ()
        self.traci.simulation.getStartingTeleportIDList.return_value = ()
        self.traci.simulation.getEndingTeleportIDList.return_value = ()
        self.traci.trafficlight.getProgram.return_value = "0"
        self.traci.trafficlight.getPhase.return_value = 0
        self.traci.trafficlight.getRedYellowGreenState.return_value = "r"
        self.traci.trafficlight.getNextSwitch.return_value = 0.0

    def step(self, time):
        self.traci.simulation.getTime.return_value = time
        self.observer.observe_step(self.traci)

    def closures(self):
        return [event for event in self.observer.store.events
                if event["event"] in ("platoon_closed", "singleton_closed")]

    def test_gap_timeout_without_another_vehicle(self):
        self.observer._add_to_platoon("a", 10.0)
        self.observer._add_to_platoon("b", 11.0)
        self.step(14.0)
        self.assertIsNotNone(self.observer.open_platoon)
        self.step(15.0)
        self.assertIsNone(self.observer.open_platoon)
        self.assertEqual(self.closures(), [{"event": "platoon_closed", "time": 15.0,
                                           "platoon_id": 1, "member_ids": ["a", "b"]}])

    def test_window_timeout_while_gap_is_still_valid(self):
        for time in (0.0, 3.0, 6.0, 9.0, 12.0, 15.0):
            self.observer._add_to_platoon(str(time), time)
            self.step(time)
        self.assertIsNotNone(self.observer.open_platoon)
        self.step(16.0)
        self.assertIsNone(self.observer.open_platoon)
        self.assertEqual(self.closures()[0]["time"], 16.0)
        self.assertEqual(len(self.closures()[0]["member_ids"]), 6)

    def test_singleton_timeout(self):
        self.observer._add_to_platoon("a", 10.0)
        self.step(13.0)
        self.assertEqual(self.closures(), [])
        self.step(14.0)
        self.assertEqual(self.closures(), [{"event": "singleton_closed", "time": 14.0,
                                           "platoon_id": 1, "member_ids": ["a"]}])

    def test_repeated_steps_and_finalize_do_not_close_twice(self):
        self.observer._add_to_platoon("a", 10.0)
        for time in (14.0, 15.0, 100.0):
            self.step(time)
        self.observer.finalize(100.0)
        self.assertEqual(len(self.closures()), 1)
        self.assertEqual(len(self.observer.platoons), 1)
        self.assertTrue(self.observer.platoons[1].closed)

    def test_membership_matches_arrival_only_logic(self):
        arrivals = {0: ("b", "a"), 3: ("c",), 6: ("d",), 9: ("e",),
                    12: ("f",), 15: ("g",), 16: ("h", "i"), 19: ("j",),
                    23: ("k",), 99: ("l",), 100: ("m",)}
        legacy = ShadowForecast()
        for time, ids in arrivals.items():
            for vid in sorted(ids):
                legacy._add_to_platoon(vid, float(time))
        legacy._close_open(110.0)

        def detect(traci, vid, time, departed, teleported):
            self.observer.seen_e1.add(vid)
            self.observer._add_to_platoon(vid, time)

        self.observer._detect = mock.Mock(side_effect=detect)
        self.traci.vehicle.getRoadID.return_value = "E1"
        for time in range(111):
            self.traci.vehicle.getIDList.return_value = arrivals.get(time, ())
            self.step(float(time))
        before = {pid: group.member_ids for pid, group in legacy.platoons.items()}
        after = {pid: group.member_ids for pid, group in self.observer.platoons.items()}
        self.assertEqual(before, after)
        self.assertIsNone(self.observer.open_platoon)
        self.assertEqual(len(self.closures()), len(after))
        self.assertEqual(self.observer._detect.call_count, sum(map(len, arrivals.values())))


@unittest.skipUnless(os.environ.get("SUMO_HOME"), "SUMO_HOME is required")
class ForecastSumoTests(unittest.TestCase):
    def test_policy_environment_metrics_are_unchanged(self):
        import numpy as np
        import traci
        from sumolib import checkBinary

        # The observer itself does not require PyTorch; stub the agent import
        # so this environment check also works on machines without it.
        agent_module = types.ModuleType("policy.agent")
        agent_module.PPOAgent = object
        spec = importlib.util.spec_from_file_location("_forecast_train_test", ROOT / "policy" / "train.py")
        train = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch"),
                                          "policy.agent": agent_module}):
            spec.loader.exec_module(train)

        def run(shadow):
            traci.start([checkBinary("sumo"), "-c", str(ROOT / "configuration.sumocfg"),
                         "--no-step-log", "true"])
            try:
                contexts = train._build_intersection_contexts()
                env = train._make_env(contexts=contexts, min_green=5, max_green=45,
                                      shadow_forecast=shadow)
                observation = env.reset()
                elapsed = 0
                total_wait = 0.0
                total_reward = 0.0
                actions = []
                while elapsed < 300:
                    observation, reward, done, duration, waiting, info = env.step(
                        np.array([0.25], dtype=np.float32), max_steps=300 - elapsed
                    )
                    elapsed += duration
                    total_wait += waiting
                    total_reward += reward
                    actions.append((info["requested_duration"], info["green_executed_duration"],
                                    info["yellow_executed_duration"], env.phase_cursor))
                    if done:
                        break
                return (observation.tolist(), total_wait, total_reward,
                        env.observational_metrics(), actions)
            finally:
                traci.close()

        self.assertEqual(run(False), run(True))

    def test_observer_preserves_static_slave_trace_and_uses_live_e1_position(self):
        import traci
        from sumolib import checkBinary

        def run(observer_enabled):
            traci.start([checkBinary("sumo"), "-c", str(ROOT / "configuration.sumocfg"),
                         "--no-step-log", "true"])
            observer = ShadowForecast() if observer_enabled else None
            trace = []
            try:
                for _ in range(300):
                    traci.simulationStep()
                    if observer is not None:
                        observer.observe_step(traci)
                    trace.append(tuple(
                        (traci.trafficlight.getProgram(tls), traci.trafficlight.getPhase(tls),
                         traci.trafficlight.getRedYellowGreenState(tls),
                         traci.trafficlight.getNextSwitch(tls))
                        for tls in ("J2", "J10", "J16")
                    ))
                if observer is not None:
                    observer.finalize(float(traci.simulation.getTime()))
            finally:
                traci.close()
            return trace, observer

        plain_trace, _ = run(False)
        shadow_trace, observer = run(True)
        repeated_trace, repeated_observer = run(True)
        self.assertEqual(plain_trace, shadow_trace)
        self.assertEqual(shadow_trace, repeated_trace)
        self.assertEqual(observer.store.events, repeated_observer.store.events)
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.jsonl"
            second = Path(folder) / "second.jsonl"
            observer.store.write_jsonl(first)
            repeated_observer.store.write_jsonl(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertGreater(len(observer.followed), 0)
        self.assertGreater(len(observer.direct_e1), 0)
        self.assertEqual(len(observer.followed), len(set(observer.followed)))
        self.assertTrue(any(vehicle.lane_position > 0 for vehicle in observer.followed.values()))
        self.assertGreater(sum(vehicle.eta_by_slave["J2"] is not None
                               for vehicle in observer.followed.values()), 0)
        self.assertGreater(sum("J2" in vehicle.actual_by_slave
                               for vehicle in observer.followed.values()), 0)
        for vehicle in observer.followed.values():
            if vehicle.distance_by_slave["J2"] is not None:
                self.assertLess(vehicle.distance_by_slave["J2"], 103.06)
            if vehicle.route[-1] == "E1":
                self.assertEqual(vehicle.status_by_slave["J2"], "no_aplicable")
                self.assertNotIn("J2", vehicle.actual_by_slave)


if __name__ == "__main__":
    unittest.main()
