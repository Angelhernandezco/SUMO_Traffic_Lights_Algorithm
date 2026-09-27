"""Movement classification and read-only measurement tests."""

import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from policy.flow_metrics import Approach, FlowMetricsObserver, classify_movement, classify_origin, summarize


class FlowClassificationTests(unittest.TestCase):
    def test_straight_movements_only(self):
        route = ("-E3", "E1", "E5", "E10", "E13")
        for tls, index in (("J2", 1), ("J10", 2), ("J16", 3)):
            self.assertEqual(classify_movement(tls, route, index)[0], "priority")

    def test_turns_on_same_access_and_shared_lane_are_secondary(self):
        for tls, incoming, turn in (("J2", "E1", "E6"), ("J10", "E5", "E8"), ("J16", "E10", "E12")):
            self.assertEqual(classify_movement(tls, (incoming, turn), 0)[0], "secondary")

    def test_opposing_and_terminal_movements_are_secondary(self):
        self.assertEqual(classify_movement("J2", ("-E5", "-E1"), 0)[0], "secondary")
        self.assertEqual(classify_movement("J2", ("E1",), 0)[0], "secondary")
        self.assertEqual(classify_movement("J2", (), -1)[0], "unknown")

    def test_j0_direct_insertions_and_side_entries_are_separated(self):
        route = ("-E0", "E1", "E5", "E10", "E13")
        self.assertEqual(classify_origin("J16", route, 3, True, False), "j0_released")
        self.assertEqual(classify_origin("J16", route[1:], 2, False, True), "direct_e1")
        self.assertEqual(classify_origin("J10", ("-E6", "E5", "E10"), 1, False, False), "other_upstream")

    def test_returning_j0_vehicle_without_contiguous_prefix_not_in_cohort(self):
        self.assertEqual(classify_origin("J16", ("E1", "E6", "E10", "E13"), 2, True, False), "other_upstream")

    def test_route_index_not_first_matching_edge(self):
        route = ("E1", "E6", "E1", "E5")
        self.assertEqual(classify_movement("J2", route, 0)[0], "secondary")
        self.assertEqual(classify_movement("J2", route, 2)[0], "priority")


class FlowObserverTests(unittest.TestCase):
    def setUp(self):
        self.traci = mock.Mock()
        self.now = 0
        self.road = "E1"
        self.lane = "E1_0"
        self.speed = 0
        self.traci.simulation.getTime.side_effect = lambda: self.now
        self.traci.simulation.getDeltaT.return_value = 1
        self.traci.simulation.getStartingTeleportIDList.return_value = []
        self.traci.simulation.getEndingTeleportIDList.return_value = []
        self.traci.vehicle.getIDList.return_value = ["a"]
        self.traci.vehicle.getRoadID.side_effect = lambda vid: self.road
        self.traci.vehicle.getRoute.return_value = ("E1", "E5", "E10", "E13")
        self.traci.vehicle.getRouteIndex.return_value = 0
        self.traci.vehicle.getSpeed.side_effect = lambda vid: self.speed
        self.traci.lane.getLastStepVehicleIDs.side_effect = lambda lane: ["a"] if lane == self.lane else []
        self.traci.lane.getLastStepHaltingNumber.side_effect = lambda lane: int(lane == self.lane and self.speed < 0.1)
        contexts = {tls: SimpleNamespace(controlled_lanes=[edge + "_0", edge + "_1"])
                    for tls, edge in (("J2", "E1"), ("J10", "E5"), ("J16", "E10"))}
        self.observer = FlowMetricsObserver(contexts)
        self.source = SimpleNamespace(followed={"a": SimpleNamespace(platoon_id=7, detected_at=1)}, direct_e1=set())
        self.local = SimpleNamespace(latest={tls: {} for tls in contexts})

    def step(self):
        self.now += 1
        self.observer.observe(self.traci, self.source, self.local, None)

    def test_waiting_stop_episodes_and_lane_change(self):
        self.step()
        self.step()
        self.lane = "E1_1"
        self.step()
        record = self.observer.active["a"]
        self.assertEqual((record.waiting, record.stops), (3, 1))
        self.speed = 5
        self.step()
        self.speed = 0
        self.step()
        self.assertEqual((record.waiting, record.stops), (4, 2))

    def test_observed_crossing_and_censored_denominator(self):
        self.speed = 5
        self.step()
        self.road, self.lane = ":J2_6", ":J2_6_0"
        self.step()
        row = self.observer.summary()["J2"]["priority"]
        self.assertEqual((row["throughput"], row["no_stop_crossing_percent"]), (1, 100))
        self.assertEqual(row["mean_j0_to_crossing_seconds"], 1)
        record = next(iter(self.observer.records.values()))
        unfinished = replace(record, vehicle_id="b", crossed_at=None)
        row = summarize([record, unfinished])
        self.assertEqual(row["vehicles"], 2)
        self.assertEqual(row["complete_crossings"], 1)
        self.assertEqual(row["unfinished_or_censored"], 1)

    def test_teleport_or_disappearance_not_a_crossing(self):
        self.step()
        self.traci.simulation.getStartingTeleportIDList.return_value = ["a"]
        self.road, self.lane = "E5", "E5_0"
        self.traci.vehicle.getRouteIndex.return_value = 1
        self.step()
        self.assertEqual(self.observer.summary()["J2"]["priority"]["throughput"], 0)

    def test_remanent_is_temporal_subset_not_extra_population(self):
        self.local.latest["J2"][7] = SimpleNamespace(source_closed=True, source_member_count=2, valid_member_ids=("a",))
        self.step()
        self.local.latest["J2"] = {}
        self.step()
        row = self.observer.summary()["J2"]["priority"]
        self.assertEqual((row["vehicles"], row["waiting"], row["remanent_waiting"]), (1, 2, 1))

    def test_live_classification_ignores_actual_arrival_labels(self):
        self.source.followed["a"].actual_by_slave = {"J2": -999}
        self.source.followed["a"].status_by_slave = {"J2": "cruzado"}
        self.step()
        self.assertEqual(self.observer.summary()["J2"]["priority"]["vehicles"], 1)

    def test_no_write_or_extra_simulation_step_and_no_source_mutation(self):
        before = copy.deepcopy(vars(self.source))
        self.step()
        self.assertEqual(vars(self.source), before)
        self.assertTrue(all(call[0].split(".")[-1].startswith("get") for call in self.traci.mock_calls))

    def test_double_step_or_missing_step_rejected(self):
        self.step()
        with self.assertRaisesRegex(ValueError, "existing simulation step"):
            self.observer.observe(self.traci, self.source, self.local, None)


class CorridorCohortTests(unittest.TestCase):
    def record(self, tls="J16", outgoing="E12", origin="j0_released", crossed_at=10):
        incoming = {"J2": "E1", "J10": "E5", "J16": "E10"}[tls]
        group = classify_movement(tls, (incoming, outgoing), 0)[0]
        return Approach(tls, "a", 0, (incoming, outgoing), group, incoming, outgoing,
                        origin, 1, incoming + "_0", 1, 1, crossed_at=crossed_at)

    def test_j16_turns_included_but_turns_at_j2_not_included(self):
        from report_flow_metrics import corridor_from_j0
        self.assertTrue(corridor_from_j0(self.record()))
        self.assertFalse(corridor_from_j0(self.record(tls="J2", outgoing="E6")))

    def test_direct_insertions_excluded(self):
        from report_flow_metrics import corridor_from_j0
        self.assertFalse(corridor_from_j0(self.record(origin="direct_e1")))

    def test_terminal_route_on_e10_not_a_j16_movement(self):
        from report_flow_metrics import corridor_from_j0
        self.assertFalse(corridor_from_j0(self.record(outgoing=None)))

    def test_pairing_distinguishes_present_and_crossed_in_both(self):
        from report_flow_metrics import compare_cohort
        a = self.record()
        b = replace(a, crossed_at=None, waiting=3)
        result = compare_cohort([a], [b])
        self.assertEqual(result["matched"]["J16"]["vehicles"]["baseline"], 1)
        self.assertEqual(result["matched"]["J16"]["waiting"]["absolute_difference"], 3)
        self.assertEqual(result["completed_in_both"]["J16"]["vehicles"]["baseline"], 0)
        self.assertIsNone(result["completed_in_both"]["J16"]["no_stop_crossing_percent"]["baseline"])

    def test_full_corridor_trip_sums_three_accesses_and_excludes_partial_route(self):
        from report_flow_metrics import corridor_trips, compare_trips
        route = ("-E0", "E1", "E5", "E10", "E12")
        j2 = replace(self.record(tls="J2", outgoing="E5"), route=route, route_index=1, waiting=10, stops=1)
        j10 = replace(self.record(tls="J10", outgoing="E10"), route=route, route_index=2, waiting=2, stops=1)
        j16 = replace(self.record(), route=route, route_index=3, waiting=3, stops=1)
        partial = replace(j2, vehicle_id="b", route=("E1", "E5", "E8"), route_index=0)
        records = [j2, j10, j16, partial]
        trips = corridor_trips(records)
        self.assertEqual(set(trips), {"a"})
        self.assertEqual((trips["a"]["waiting"], trips["a"]["stops"]), (15, 3))
        result = compare_trips(records, [j2, j10, replace(j16, crossed_at=None)])
        self.assertEqual(result["comparison"]["matched"]["vehicles"]["advance"], 1)
        self.assertEqual(result["comparison"]["completed_in_both"]["vehicles"]["baseline"], 0)


if __name__ == "__main__":
    unittest.main()
