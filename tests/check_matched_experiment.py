"""Unit checks for matched measurements and recovery; no SUMO simulations."""
import copy
import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sync.common import ROOT, resolve_demand
from tools.run_matched_experiment import (REFERENCE, SIX_CSV, begin_attempt, check_row, digest,
                                          finish_attempt, measured_row, open_state, route_content,
                                          cohorts, percentage, export, validate_references)


class MatchedExperimentTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "results").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.route = self.directory / "demand.rou.xml"
        self.route.write_text('<routes><vehicle id="a" depart="0"><route edges="-E0 E1 E5 E10 E13"/></vehicle>'
                              '<vehicle id="b" depart="2"><route edges="-E0 E1 E5 E10 E13"/></vehicle></routes>')
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        self.summary = copy.deepcopy(reference["runs"]["seed42_v3_full"]["summary"])
        s = self.summary
        s.update(cohort=2, crossed=1, no_stop=1, no_stop_pct=50.0, waiting_E1_s=3,
                 mean_E1_to_E5_s=12, departed=2, arrived=1,
                 pending={"min_expected": 1, "active": 1, "pending_insertion": 0})
        for key in ("full_corridor_any_exit", "full_corridor_straight_exit"):
            s[key].update(cohort=2, completed=1, pending=1, no_stop_all_three=1,
                          no_stop_all_three_pct=50.0, waiting_total_s=10,
                          J0_to_J16_s={"n": 1, "mean": 32, "median": 32, "p90": 32})
        self.vehicles = [{"vehicle_id": "a", "exit_J16": "E13", "J2_entered_at": 1, "J2_waiting_s": 3},
                         {"vehicle_id": "b", "exit_J16": "E13", "J2_entered_at": 2, "J2_waiting_s": 7}]

    def result(self):
        return (self.summary, [], [], [], [{"time": 3600}], [], [], self.vehicles)

    def test_pending_wait_and_both_denominators(self):
        row = measured_row(self.result(), self.route)
        self.assertEqual(row["waiting_E1_s"], 3)
        self.assertEqual(row["waiting_E1_all_observed_s"], 10)
        self.assertEqual(row["cohort_J2_observed_E1"], 2)
        for suffix in ("J2", "full"):
            self.assertEqual(row[f"cohort_{suffix}"], 2)
            self.assertEqual(row[f"completed_{suffix}"], 1)
            self.assertEqual(row[f"no_stop_{suffix}_pct"], 50)
            self.assertEqual(row[f"no_stop_{suffix}_completed_pct"], 100)

    def test_all_completed_waits_match(self):
        self.summary.update(crossed=2, no_stop_pct=50, waiting_E1_s=10)
        row = measured_row(self.result(), self.route)
        self.assertEqual(row["waiting_E1_all_observed_s"], row["waiting_E1_s"])

    def test_no_completed_samples(self):
        self.summary.update(crossed=0, no_stop=0, no_stop_pct=0, waiting_E1_s=0, mean_E1_to_E5_s=None)
        row = measured_row(self.result(), self.route)
        self.assertEqual(row["no_stop_J2_pct"], 0)
        self.assertIsNone(row["no_stop_J2_completed_pct"])
        self.assertIsNone(row["mean_E1_to_E5_s"])
        self.assertEqual(row["waiting_E1_all_observed_s"], 10)

    def test_empty_cohort_and_no_observation(self):
        self.route.write_text('<routes/>')
        self.vehicles.clear()
        self.summary.update(cohort=0, crossed=0, no_stop=0, no_stop_pct=None, waiting_E1_s=0,
                            mean_E1_to_E5_s=None)
        for key in ("full_corridor_any_exit", "full_corridor_straight_exit"):
            self.summary[key].update(cohort=0, completed=0, pending=0, no_stop_all_three=0,
                                     no_stop_all_three_pct=None, stops_per_vehicle=None,
                                     J0_to_J16_s={"n": 0})
        row = measured_row(self.result(), self.route)
        self.assertEqual(row["cohort_J2_observed_E1"], 0)
        self.assertIsNone(row["waiting_E1_all_observed_s"])
        self.assertIsNone(row["waiting_E1_s"])
        self.assertIsNone(row["full_waiting_s"])
        self.assertIsNone(row["no_stop_full_completed_pct"])

    def test_observed_zero_is_a_real_zero(self):
        self.summary["waiting_E1_s"] = 0
        for v in self.vehicles:
            v["J2_waiting_s"] = 0
        self.assertEqual(measured_row(self.result(), self.route)["waiting_E1_all_observed_s"], 0)

    def test_unobserved_members_do_not_impute_wait(self):
        self.vehicles[1]["J2_entered_at"] = None
        self.vehicles[1]["J2_waiting_s"] = 0
        row = measured_row(self.result(), self.route)
        self.assertEqual(row["cohort_J2_observed_E1"], 1)
        self.assertEqual(row["waiting_E1_all_observed_s"], 3)

    def test_invalid_percent_and_wait_are_rejected(self):
        row = measured_row(self.result(), self.route)
        row["no_stop_full_completed_pct"] = 50
        with self.assertRaises(ValueError):
            check_row(row)
        row = measured_row(self.result(), self.route)
        row["waiting_E1_all_observed_s"] = 2
        with self.assertRaises(ValueError):
            check_row(row)

    def test_reference_xml_comments_are_not_demand(self):
        original = route_content(self.route)
        self.route.write_text(self.route.read_text().replace('<routes>', '<routes><!--different date-->'))
        self.assertEqual(route_content(self.route), original)
        self.route.write_text(self.route.read_text().replace('depart="2"', 'depart="3"'))
        self.assertNotEqual(route_content(self.route), original)

    def test_programmed_cohorts_are_independent_of_completion(self):
        before = cohorts(self.route)
        self.summary["crossed"] = 0
        self.assertEqual(cohorts(self.route), before)
        late = self.route.read_text().replace('depart="2"', 'depart="3421"')
        self.route.write_text(late)
        self.assertEqual(cohorts(self.route)["J2"], {"a"})

    def test_route_resolver_preserves_legacy_and_requires_explicit_new(self):
        self.assertEqual(resolve_demand(42), ROOT / "sync/demand/seed_42.rou.xml")
        with self.assertRaises(ValueError):
            resolve_demand(1001)
        self.assertEqual(resolve_demand(1001, self.route), self.route.resolve())
        with self.assertRaises(FileNotFoundError):
            resolve_demand(1001, self.directory / "missing.xml")

    def test_resume_failure_retry_no_duplicate_and_hash_rejection(self):
        manifest = {"experiment_id": "one"}
        path = self.directory / "runs.sqlite"
        db = open_state(path, manifest)
        self.addCleanup(db.close)
        self.assertTrue(begin_attempt(db, "one", "bajo", 1001, "v3"))
        db.close()  # Simulate interruption while status=running.
        db = open_state(path, manifest)
        self.assertTrue(begin_attempt(db, "one", "bajo", 1001, "v3"))
        finish_attempt(db, "one", "bajo", 1001, "v3", {"result": 1}, {"signature": "x"})
        self.assertFalse(begin_attempt(db, "one", "bajo", 1001, "v3"))
        self.assertEqual(db.execute("SELECT count(*) FROM runs").fetchone()[0], 1)
        self.assertTrue(begin_attempt(db, "one", "bajo", 1002, "v3"))
        finish_attempt(db, "one", "bajo", 1002, "v3", error="failure 1")
        self.assertTrue(begin_attempt(db, "one", "bajo", 1002, "v3"))
        finish_attempt(db, "one", "bajo", 1002, "v3", error="failure 2")
        self.assertFalse(begin_attempt(db, "one", "bajo", 1002, "v3"))
        db.close()
        with self.assertRaises(ValueError):
            open_state(path, {"experiment_id": "changed"})

    def test_teleports_are_kept_and_flagged(self):
        self.summary["teleported"] = 1
        row = measured_row(self.result(), self.route)
        self.assertEqual(row["teleports"], 1)
        self.assertIn("teleports_observed", row["quality_flags"])

    def test_empty_denominator(self):
        self.assertIsNone(percentage(0, 0))

    def test_export_is_atomic_and_pairing_checks_programmed_not_completed(self):
        manifest = {"experiment_id": "one", "resource_sha256": {}}
        db = open_state(self.directory / "runs.sqlite", manifest)
        self.addCleanup(db.close)
        row = measured_row(self.result(), self.route)
        row.update(traffic_level="bajo", seed=1001, run_id="one:bajo:1001:v3")
        begin_attempt(db, "one", "bajo", 1001, "v3")
        finish_attempt(db, "one", "bajo", 1001, "v3", row)
        with patch("tools.run_matched_experiment.snapshot", return_value={}):
            self.assertFalse(export(self.directory, db, manifest, {}))
            self.assertFalse(export(self.directory, db, manifest, {}))
        with (self.directory / "per_demand.csv").open(encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)
        second = dict(row, mode="plain", completed_J2=2, no_stop_J2_completed_pct=50,
                      run_id="one:bajo:1001:plain")
        begin_attempt(db, "one", "bajo", 1001, "plain")
        finish_attempt(db, "one", "bajo", 1001, "plain", second)
        with patch("tools.run_matched_experiment.snapshot", return_value={}):
            export(self.directory, db, manifest, {})
            second["cohort_J2_sha256"] = "different"
            finish_attempt(db, "one", "bajo", 1001, "plain", second)
            with self.assertRaises(ValueError):
                export(self.directory, db, manifest, {})

    def test_historical_validation_does_not_require_results(self):
        frozen = json.loads(REFERENCE.read_text(encoding="utf-8"))
        with SIX_CSV.open(encoding="utf-8") as handle:
            saved = {r["mode"]: r for r in csv.DictReader(handle) if r["seed"] == "42"}
        payloads = []
        for mode in ("plain", "plain_offset", "only_ppo", "v3"):
            row = {}
            for key, value in saved[mode].items():
                try:
                    row[key] = json.loads(value) if value else None
                except json.JSONDecodeError:
                    row[key] = {"True": True, "False": False}.get(value, value)
            baseline = frozen["runs"]["seed42_v3_full"]
            payloads.append({"historical_row": row, "summary": baseline["summary"],
                             "canonical_traces": baseline["traces"]})
        with patch("tools.run_matched_experiment.ROOT", self.directory), \
             patch("tools.run_matched_experiment.generate", return_value={}), \
             patch("tools.run_matched_experiment.route_content", return_value=[]), \
             patch("tools.run_matched_experiment.execute_worker", side_effect=payloads):
            evidence = validate_references(self.directory)
        self.assertEqual(len(evidence["historical_runs"]), 4)
        self.assertTrue(all(all(source.startswith("analysis/") for source in r["sources"])
                            for r in evidence["historical_runs"]))


if __name__ == "__main__":
    unittest.main()
