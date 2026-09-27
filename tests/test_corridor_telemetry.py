"""Causal event and read-only boundary checks for pre-C2 telemetry."""

import copy
import unittest
from types import SimpleNamespace
from unittest import mock

from policy.corridor_telemetry import CorridorTelemetry, ReadOnlyTraCI, temporal_spread


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.raw = mock.Mock()
        self.raw.TraCIException = RuntimeError
        self.api = ReadOnlyTraCI(self.raw)
        self.now, self.road, self.speed = 0, "E1", 0.0
        self.raw.simulation.getTime.side_effect = lambda: self.now
        self.raw.simulation.getDeltaT.return_value = 1
        self.raw.simulation.getStartingTeleportIDList.return_value = []
        self.raw.simulation.getEndingTeleportIDList.return_value = []
        self.raw.simulation.getArrivedIDList.return_value = []
        self.raw.vehicle.getIDList.return_value = ["a"]
        self.raw.vehicle.getRoadID.side_effect = lambda vid: self.road
        self.raw.vehicle.getRoute.return_value = ("-E0", "E1", "E5", "E10", "E13")
        self.raw.vehicle.getRouteIndex.side_effect = lambda vid: 2 if self.road == "E5" else 1
        self.raw.vehicle.getLaneID.side_effect = lambda vid: self.road + "_0"
        self.raw.vehicle.getLanePosition.return_value = 25.0
        self.raw.vehicle.getSpeed.side_effect = lambda vid: self.speed
        self.raw.vehicle.getLaneIndex.return_value = 0
        self.raw.vehicle.getAllowedSpeed.return_value = 10
        self.raw.vehicle.getDrivingDistance.return_value = 60
        self.raw.edge.getLaneNumber.return_value = 1
        self.raw.lane.getLength.return_value = 100
        self.raw.lane.getMaxSpeed.return_value = 10
        self.raw.lane.getLastStepVehicleIDs.return_value = ("a",)
        self.raw.lane.getLastStepOccupancy.return_value = 30.0
        self.raw.trafficlight.getPhase.side_effect = lambda tls: 2 if self.now >= 2 else 0
        self.raw.trafficlight.getRedYellowGreenState.side_effect = lambda tls: "G" if self.now >= 2 else "r"
        self.raw.trafficlight.getNextSwitch.return_value = 10.0
        def links(tls):
            edge, outgoing = {"J2": ("E1", "E5"), "J10": ("E5", "E10"), "J16": ("E10", "E13")}[tls]
            return [[(edge + "_0", outgoing + "_0", ":internal")]]
        self.raw.trafficlight.getControlledLinks.side_effect = links
        vehicle = SimpleNamespace(detected_at=1., platoon_id=7,
                                  route=("-E0", "E1", "E5", "E10", "E13"),
                                  eta_by_slave={"J2": 7.0}, censor_reason=None)
        self.source = SimpleNamespace(followed={"a": vehicle}, direct_e1=set(),
                                      platoons={7: SimpleNamespace(member_ids=["a"], closed=True)})
        self.observer = CorridorTelemetry()

    def step(self, road=None, speed=None):
        self.now += 1
        if road is not None:
            self.road = road
        if speed is not None:
            self.speed = speed
        self.observer.observe(self.api, self.source)

    def test_samples_stop_cross_exit_and_segment(self):
        before = copy.deepcopy(vars(self.source.followed["a"]))
        self.step()
        self.step(speed=6)
        self.step(road=":J2_0", speed=6)
        self.step(road="E5", speed=6)
        events = [e["event"] for e in self.observer.events]
        for kind in ("j0_release_observed", "approach_enter", "stop_begin", "stop_end",
                     "stop_line_cross", "intersection_exit", "next_link_enter"):
            self.assertIn(kind, events)
        self.assertEqual(self.observer.visits["J2:a:1"]["waiting"], 1)
        sample = self.observer.samples[0]
        self.assertEqual((sample["tls_id"], sample["movement_green"], sample["movement_signal"]), ("J2", False, "r"))
        self.assertEqual(sample["distance_to_stop_line"], 75)
        self.assertEqual(sample["downstream_occupancy_percent"], {"E5_0": 30.0})
        self.assertEqual(self.observer.export(self.source)["segments"][0]["cross_seconds"], 2)
        self.assertEqual(vars(self.source.followed["a"]), before)
        self.assertTrue(all(call[0].split(".")[-1].startswith("get") for call in self.raw.mock_calls))
        with self.assertRaisesRegex(AssertionError, "cannot invoke"):
            self.api.simulationStep()

    def test_censored_reappearance_never_creates_crossing(self):
        self.step()
        self.raw.simulation.getStartingTeleportIDList.return_value = ["a"]
        self.step(road=":J2_0", speed=6)
        self.raw.simulation.getStartingTeleportIDList.return_value = []
        self.step(road="E5", speed=6)
        self.assertEqual(self.observer.censored["a"], "teleport")
        self.assertFalse(any(e["event"] == "stop_line_cross" for e in self.observer.events))
        self.assertEqual(self.observer.export(self.source)["segments"][0]["complete"], False)

    def test_group_temporal_compactness_requires_two_observations(self):
        self.assertIsNone(temporal_spread([1])["compact_observed_subset"])
        self.assertTrue(temporal_spread([1, 3, 6])["compact_observed_subset"])
        self.assertFalse(temporal_spread([1, 5])["compact_observed_subset"])

    def test_no_duplicate_observation_of_same_simulation_step(self):
        self.step()
        with self.assertRaisesRegex(ValueError, "existing simulation step"):
            self.observer.observe(self.api, self.source)


if __name__ == "__main__":
    unittest.main()
