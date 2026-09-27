"""Causal forecasts and hypothetical ledgers, without a slave executor."""

import copy
import json
import io
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from policy.coordination import (
    LocalForecast, LocalForecastObserver, NOMINAL_DURATIONS, SignalSnapshot,
    SlaveProgram, ShadowAdvanceEvaluator, TLS_ORDER, read_programs, validate_modes,
)


def programs():
    return {tls: SlaveProgram(tls, "0", tuple(map(float, durations)),
                             tuple("Grr" if i % 2 == 0 else "yrr" for i in range(len(durations))),
                             frozenset({(edge, outgoing)}))
            for tls, edge, outgoing, durations in (
                ("J2", "E1", "E5", NOMINAL_DURATIONS["J2"]),
                ("J10", "E5", "E10", NOMINAL_DURATIONS["J10"]),
                ("J16", "E10", "E13", NOMINAL_DURATIONS["J16"]))}


def signals():
    return {tls: SignalSnapshot("0", 0, "Grr", p.durations[0], 0.0)
            for tls, p in programs().items()}


def packet(pid=1, now=1.0, ready=2.0, **changes):
    result = LocalForecast("J2", pid, 1, now, True, 2, ("a",), ("a",),
                           1, ready, ready, "complete", True)
    return replace(result, **changes)


def forecasts(*packets):
    result = {tls: {} for tls in TLS_ORDER}
    for p in packets:
        result[p.tls_id][p.platoon_id] = p
    return result


class LocalForecastTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.observer = LocalForecastObserver(programs(),
                                             lambda event, time, **data: self.events.append((event, time, data)))
        route = ("-E0", "E1", "E5", "E10", "E13")
        self.source = SimpleNamespace(
            previous_roads={"a": "E1", "b": "E1", "direct": "E1"},
            followed={vid: SimpleNamespace(route=route, platoon_id=1,
                                           actual_by_slave={"J2": 999},
                                           status_by_slave={"J2": "cruzado"}, censor_reason="teleport")
                      for vid in ("a", "b")},
            platoons={1: SimpleNamespace(closed=True, member_ids=["a", "b"])})
        self.traci = mock.Mock()
        self.traci.TraCIException = RuntimeError
        self.traci.vehicle.getRoute.side_effect = lambda vid: self.source.followed[vid].route
        self.traci.vehicle.getRouteIndex.side_effect = lambda vid: self.source.followed[vid].route.index(
            self.source.previous_roads[vid])
        self.traci.vehicle.getLaneID.side_effect = lambda vid: self.source.previous_roads[vid] + "_0"
        self.traci.vehicle.getLanePosition.side_effect = lambda vid: {"a": 30, "b": 50}[vid]
        self.traci.vehicle.getAllowedSpeed.return_value = 10
        self.traci.lane.getLength.return_value = 100
        self.traci.lane.getMaxSpeed.return_value = 10

    def observe(self, time=10):
        return self.observer.observe(self.traci, self.source, float(time))

    def test_live_distance_revisions_and_direct_insertions(self):
        first = self.observe()["J2"][1]
        self.assertEqual(first.member_ids, ("a", "b"))
        self.assertEqual((first.ready_earliest, first.ready_latest), (15, 17))
        self.traci.vehicle.getLanePosition.side_effect = lambda vid: 90
        second = self.observe(11)["J2"][1]
        self.assertEqual(second.revision, 2)
        self.assertEqual(second.ready_earliest, 12)
        self.assertEqual(second.observed_at, 11)
        self.assertNotIn("direct", second.member_ids)
        self.assertNotIn("actual_arrival", json.dumps(self.events))

    def test_remanent_and_retirement_use_live_roads(self):
        self.observe()
        self.source.previous_roads = {"b": "E1"}
        remaining = self.observe(11)["J2"][1]
        self.assertEqual(remaining.valid_member_ids, ("b",))
        self.assertEqual(remaining.source_member_count, 2)
        self.source.previous_roads = {}
        self.assertEqual(self.observe(12)["J2"], {})
        self.assertEqual(self.events[-1][2]["reason"], "no_local_members")
        self.assertEqual(self.observer.summary()["J2"]["valid_member_coverage"], 1)

    def test_partial_and_unavailable_do_not_reuse_old_eta(self):
        self.observe()
        def speed(vid):
            if vid == "b":
                raise RuntimeError("unavailable")
            return 10
        self.traci.vehicle.getAllowedSpeed.side_effect = speed
        partial = self.observe(11)["J2"][1]
        self.assertEqual(partial.quality, "partial")
        self.assertEqual(partial.valid_member_ids, ("a",))
        self.assertEqual(partial.eligible_member_count, 2)
        self.assertEqual(partial.member_failures, (("b", "data_unavailable"),))
        self.traci.vehicle.getAllowedSpeed.side_effect = RuntimeError("unavailable")
        empty = self.observe(12)["J2"][1]
        self.assertIsNone(empty.ready_earliest)
        self.assertEqual(empty.valid_member_ids, ())
        self.assertEqual(empty.quality, "unavailable")
        self.assertEqual(empty.eligible_member_count, 2)
        stats = self.observer.summary()["J2"]
        self.assertEqual(stats["valid_member_coverage"], 3 / 6)

    def test_failed_route_queries_remain_visible_in_coverage(self):
        self.traci.vehicle.getRoute.side_effect = RuntimeError("unavailable")
        result = self.observe()["J2"][1]
        self.assertEqual(result.eligible_member_count, 2)
        self.assertEqual(self.observer.summary()["J2"]["valid_member_coverage"], 0)

    def test_incompatible_and_terminal_routes(self):
        for vid in ("a", "b"):
            self.source.followed[vid].route = ("-E0", "E1")
        result = self.observe()["J2"][1]
        self.assertFalse(result.compatible)
        self.assertEqual(result.member_failures, (("a", "incompatible"), ("b", "incompatible")))
        self.assertIsNone(self.observer.summary()["J2"]["valid_member_coverage"])

    def test_arrival_labels_cannot_change_local_views(self):
        first = self.observe()["J2"][1]
        self.observer = LocalForecastObserver(programs(), lambda *args, **kwargs: None)
        for vehicle in self.source.followed.values():
            vehicle.actual_by_slave = {"J2": -999}
            vehicle.status_by_slave = {"J2": "pendiente"}
            vehicle.censor_reason = None
        self.assertEqual(first, self.observe()["J2"][1])


class ShadowLedgerTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.evaluator = ShadowAdvanceEvaluator(programs(), 1, 2000,
                                               lambda event, time, **data: self.events.append(data))
        self.signals = signals()
        self.evaluator.seed_phases(self.signals)

    def evaluate(self, now=1, *packets):
        self.evaluator.evaluate(now, self.signals, forecasts(*(packets or (packet(now=now),))))
        return [event for event in self.events if event["tls_id"] == "J2"][-1]

    def test_only_one_reservation_and_no_revision_renewal(self):
        before = copy.deepcopy(self.signals)
        first = self.evaluate()
        self.assertEqual(first["result"], "shadow_reserved")
        self.assertEqual(first["hypothetical_reduction"], 5)
        second = self.evaluate(2, packet(now=2, ready=3, revision=2))
        self.assertIn("budget_reserved", second["failed_conditions"])
        self.assertTrue(second["geometric_candidate"])
        self.assertEqual(self.evaluator.summary()["J2"]["geometric_pairs"], 1)
        self.assertEqual(self.signals, before)

    def test_cooldown_through_full_cyclic_sequence(self):
        self.evaluate()
        p = programs()["J2"]
        start = 0.0
        for cycle in range(2):
            for phase, duration in enumerate(p.durations):
                if cycle == 0 and phase == 0:
                    start += duration
                    continue
                self.signals["J2"] = SignalSnapshot("0", phase, p.states[phase], start + duration, start)
                active = (packet(pid=2, now=start + 1, ready=start + 2),) if phase == 4 else ()
                self.evaluator.evaluate(start + 1, self.signals, forecasts(*active))
                if phase == 4:
                    event = [e for e in self.events if e["tls_id"] == "J2"][-1]
                    if cycle == 0:
                        self.assertEqual(event["reason"], "cooldown")
                    else:
                        self.assertEqual(event["result"], "shadow_reserved")
                start += duration
        self.assertEqual([r["occurrence"] for r in self.evaluator.reservations["J2"]], [1, 3])

    def test_deterministic_conflict_tiebreak_and_geometry_count(self):
        self.evaluate(1, packet(pid=2), packet(pid=1))
        events = [e for e in self.events if e["tls_id"] == "J2"]
        self.assertEqual(events[0]["platoon_id"], 1)
        self.assertEqual(events[0]["result"], "shadow_reserved")
        self.assertEqual(events[1]["reason"], "conflict_selected_other")
        self.assertEqual(self.evaluator.summary()["J2"]["geometric_pairs"], 2)

    def test_stale_and_future_views_never_reserve(self):
        for observed_at, expected in ((-1, "stale"), (2, "future_observation")):
            event = self.evaluate(1, packet(observed_at=observed_at))
            self.assertIn(expected, event["failed_conditions"])
        self.assertEqual(self.evaluator.reservations["J2"], [])

    def test_all_failed_conditions_are_logged(self):
        event = self.evaluate(1, packet(source_closed=False, valid_member_ids=(),
                                       compatible=False, ready_earliest=None, quality="unavailable"))
        self.assertEqual(event["failed_conditions"], ["platoon_open", "data_unavailable", "incompatible"])
        self.assertEqual(event["reason"], "platoon_open")

    def test_singletons_and_horizon(self):
        event = self.evaluate(1, packet(source_member_count=1))
        self.assertEqual(event["reason"], "singleton_original")
        self.evaluator.horizon = 13
        event = self.evaluate(1, packet())
        self.assertTrue(event["geometric_candidate"])
        self.assertEqual(event["reason"], "opening_outside_horizon")
        self.assertEqual(self.evaluator.summary()["J2"]["pairs_within_horizon_before_budget"], 0)

    def test_yellow_and_receptor_do_not_reserve(self):
        self.signals["J2"] = SignalSnapshot("0", 1, "yrr", 19, 15)
        event = self.evaluate(16, packet(now=16, ready=17))
        self.assertIn("yellow", event["failed_conditions"])
        self.signals["J2"] = SignalSnapshot("0", 2, "Grr", 34, 19)
        event = self.evaluate(20, packet(now=20, ready=21))
        self.assertIn("receptor_green", event["failed_conditions"])
        self.assertEqual(self.evaluator.reservations["J2"], [])

    def test_minimum_limit_lookahead_and_delay(self):
        self.signals["J2"] = replace(self.signals["J2"], next_switch=10)
        event = self.evaluate(1, packet(ready=20))
        self.assertIn("minimum_green_limit", event["failed_conditions"])
        self.assertIn("availability_outside_lookahead", event["failed_conditions"])
        self.assertIn("insufficient_advance", event["failed_conditions"])

    def test_noncyclic_transition_is_rejected(self):
        self.signals["J2"] = SignalSnapshot("0", 4, "Grr", 53, 38)
        with self.assertRaisesRegex(ValueError, "Noncyclic"):
            self.evaluate(39)

    def test_empty_tls_has_one_abstention_each_step(self):
        self.evaluator.evaluate(1, self.signals, forecasts())
        self.assertEqual(len(self.events), 3)
        self.assertTrue(all(e["reason"] == "no_forecast" for e in self.events))


class InterfaceTests(unittest.TestCase):
    def test_c1_is_not_a_supported_mode(self):
        for eta, coordination in (("baseline", "shadow"), ("local", "advance"), ("unknown", "off")):
            with self.assertRaises(ValueError):
                validate_modes(eta, coordination)
        for eta, coordination in (("baseline", "off"), ("local", "off"), ("local", "shadow")):
            validate_modes(eta, coordination)

    def network_mock(self):
        root = ET.parse(Path(__file__).resolve().parents[1] / "maps/master_slave.net.xml").getroot()
        logics, links = {}, {}
        for tls in TLS_ORDER:
            xml_logic = next(p for p in root.findall("tlLogic") if p.get("id") == tls)
            logics[tls] = SimpleNamespace(programID=xml_logic.get("programID"), type=0,
                phases=[SimpleNamespace(duration=float(p.get("duration")), state=p.get("state"), next=())
                        for p in xml_logic.findall("phase")])
            links[tls] = [[] for _ in logics[tls].phases[0].state]
            for link in root.findall("connection"):
                if link.get("tl") == tls:
                    links[tls][int(link.get("linkIndex"))].append((
                        link.get("from") + "_" + link.get("fromLane"),
                        link.get("to") + "_" + link.get("toLane"), link.get("via")))
        traci = mock.Mock()
        traci.trafficlight.getProgram.side_effect = lambda tls: logics[tls].programID
        traci.trafficlight.getAllProgramLogics.side_effect = lambda tls: [logics[tls]]
        traci.trafficlight.getControlledLinks.side_effect = lambda tls: links[tls]
        return traci, logics

    def test_real_network_mapping_and_j2_yellow_are_preserved(self):
        traci, _ = self.network_mock()
        result = read_programs(traci)
        self.assertEqual(result["J2"].durations[5], 24)
        self.assertIn(("E1", "E5"), result["J2"].movements)
        self.assertIn(("E5", "E10"), result["J10"].movements)
        self.assertIn(("E10", "E13"), result["J16"].movements)
        self.assertFalse(any(call[0].split(".")[-1].startswith("set") for call in traci.mock_calls))

    def test_changed_yellow_branching_and_incompatible_receptor_fail_startup(self):
        for mutation in ("yellow", "branch", "receptor"):
            traci, logics = self.network_mock()
            if mutation == "yellow":
                logics["J2"].phases[5].duration = 4
            elif mutation == "branch":
                logics["J2"].phases[0].next = (2,)
            else:
                state = logics["J2"].phases[2].state
                logics["J2"].phases[2].state = state[:6] + "r" + state[7:]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                read_programs(traci)

    def test_cli_defaults_and_rejected_c1_or_training_modes(self):
        import main
        with mock.patch("sys.argv", ["main.py", "--policy-test"]):
            options = main.get_options()
        self.assertEqual((options.eta_mode, options.coordination_mode), ("baseline", "off"))
        for args in (("--policy-test", "--coordination-mode", "advance"),
                     ("--policy-test", "--coordination-mode", "shadow"),
                     ("--policy-train", "--eta-mode", "local"),
                     ("--policy-test", "--coordination-output", "diagnostic.jsonl")):
            with self.subTest(args=args), mock.patch("sys.argv", ["main.py", *args]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main.get_options()
                self.assertEqual(raised.exception.code, 2)
        with mock.patch("sys.argv", ["main.py", "--policy-test", "--eta-mode", "local",
                                     "--coordination-mode", "shadow", "--forecast-output", "baseline.jsonl",
                                     "--coordination-output", "c0.jsonl"]):
            options = main.get_options()
        self.assertEqual(options.coordination_output, "c0.jsonl")


if __name__ == "__main__":
    unittest.main()
