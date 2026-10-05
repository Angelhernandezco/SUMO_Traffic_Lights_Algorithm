"""Passive observations; preserve historical cohorts and 1 s sample semantics."""
import csv
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
import numpy as np
from .common import traci, tc, TLS, RECEIVERS, END, CHECKPOINT, NET, OFFSETS
from sumo_utils import get_lane_metrics

def route_ids(path, segment, max_depart=None):
    result = set()
    for vehicle in ET.parse(path).iter("vehicle"):
        if max_depart is not None and float(vehicle.get("depart")) > max_depart:
            continue
        edges = vehicle.find("route").get("edges").split()
        if any(edges[i:i + len(segment)] == segment
               for i in range(len(edges) - len(segment) + 1)):
            result.add(vehicle.get("id"))
    return result


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def distribution(values):
    values = sorted(float(x) for x in values if x is not None)
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "p90": round(float(np.percentile(values, 90)), 2),
            "min": round(values[0], 2), "max": round(values[-1], 2)}


class FlowAudit:
    """Observe J0-straight crossings and E1 arrivals without modifying control."""

    def __init__(self, route_path):
        self.j0_straight = route_ids(route_path, ["-E0", "E1"])
        self.j2_straight = route_ids(route_path, ["-E0", "E1", "E5"])
        self.eval_strict = route_ids(route_path, ["-E0", "E1", "E5"], END - 180)
        self.last_release_id = None
        self.crossings = []
        self.vehicles = {}

    def begin_release(self, observer):
        self.last_release_id = observer.release_windows[-1]["id"]

    def sample(self):
        now = int(traci.simulation.getTime())
        phase = traci.trafficlight.getPhase("J2")
        state = traci.trafficlight.getRedYellowGreenState("J2")
        for vid in traci.vehicle.getIDList():
            if vid not in self.j0_straight:
                continue
            road = traci.vehicle.getRoadID(vid)
            row = self.vehicles.setdefault(vid, {
                "vehicle_id": vid, "J2_straight": vid in self.j2_straight,
                "evaluation_cohort": vid in self.eval_strict,
                "release_id": None, "J0_cross_time": None, "J0_cross_basis": None,
                "E1_entry_time": None, "J2_arrival_time": None,
                "J2_arrival_basis": None, "J2_signal_at_arrival": None,
                "J2_signal_certainty": None, "J2_cross_time": None,
                "last_E1_signal": None, "last_E1_time": None,
                "stopped_E1": False, "E1_wait_s": 0,
            })
            if (road.startswith(":J0_6") and row["J0_cross_time"] is None
                    and self.last_release_id is not None):
                self._cross(row, now, "J0_internal")
            if road == "E1":
                if row["E1_entry_time"] is None:
                    row["E1_entry_time"] = now
                if row["J0_cross_time"] is None and self.last_release_id is not None:
                    self._cross(row, now, "E1_entry_upper_bound")
                if traci.vehicle.getSpeed(vid) < 0.1:
                    row["stopped_E1"] = True
                    row["E1_wait_s"] += 1
                tls = next((item for item in traci.vehicle.getNextTLS(vid)
                            if item[0] == "J2"), None)
                if tls:
                    row["last_E1_signal"] = state[tls[1]]
                    row["last_E1_time"] = now
                    if row["J2_arrival_time"] is None and tls[2] <= 5:
                        row.update({"J2_arrival_time": now,
                                    "J2_arrival_basis": "first_within_5m",
                                    "J2_signal_at_arrival": state[tls[1]],
                                    "J2_signal_certainty": "observed_within_5m"})
            elif road.startswith(":J2_") and row["J2_cross_time"] is None:
                row["J2_cross_time"] = now
                if row["J2_arrival_time"] is None:
                    link = int(road.split("_")[1])
                    current_signal = state[link]
                    stable = row["last_E1_signal"] == current_signal
                    row.update({"J2_arrival_time": now,
                                "J2_arrival_basis": "crossed_between_samples",
                                "J2_signal_at_arrival": (current_signal if stable else
                                                         "transition_ambiguous"),
                                "J2_signal_certainty": ("stable_across_1s" if stable else
                                                        "transition_ambiguous")})

    def _cross(self, row, now, basis):
        row["J0_cross_time"] = now
        row["J0_cross_basis"] = basis
        row["release_id"] = self.last_release_id
        self.crossings.append({"release_id": self.last_release_id,
                               "vehicle_id": row["vehicle_id"], "time": now,
                               "basis": basis})


def overlap(start, stop, green_intervals):
    return sum(max(0, min(stop, b) - max(start, a)) for a, b in green_intervals)


class Observer:
    def __init__(self, end, route):
        # Match plain.run_plain: all TLS-controlled lanes, counted only once.
        self.metric_lanes = sorted({
            lane for tls in traci.trafficlight.getIDList()
            for lane in traci.trafficlight.getControlledLanes(tls)
        })
        self.metric_delta_t = traci.simulation.getDeltaT()
        self.metric_steps = 0
        self.metric_waiting_time = 0
        self.metric_effective_flow = 0.0
        self.metric_queue_sum = 0.0
        self.cohort = route_ids(route, ["-E0", "E1", "E5"], end - 180)
        self.broad_cohort = route_ids(route, ["E1", "E5", "E10"], end - 180)
        self.records = {}
        self.J0_releases = []
        self.release_windows = []
        self.active_release = None
        incoming = {lane.rsplit("_", 1)[0]
                    for lane in traci.trafficlight.getControlledLanes("J2")}
        self.secondary_edges = sorted(incoming - {"E1"})
        self.secondary_wait = 0
        self.J2_phase_counts = {i: 0 for i in range(8)}
        self.last_J2_phase = traci.trafficlight.getPhase("J2")
        self.J2_phase_ticks = 0
        self.J2_phase_runs = []
        self.J10_phase_starts = []
        self.J16_phase_starts = []
        self.last_other_phase = {tls: traci.trafficlight.getPhase(tls)
                                 for tls in ("J10", "J16")}

    def begin_release(self, start, green):
        release = {"id": len(self.release_windows), "start": start,
                   "end": start + green, "E1_entries": []}
        self.release_windows.append(release)
        self.active_release = release

    def end_release(self):
        self.active_release = None

    def finish(self):
        self.J2_partial_run = {"phase": self.last_J2_phase,
                               "duration": self.J2_phase_ticks,
                               "ended_at": int(traci.simulation.getTime()) + 1,
                               "partial": True}

    def lane_metrics(self):
        """Return the same three aggregates and units as plain.run_plain."""
        return {
            "waiting_time": self.metric_waiting_time,
            "effective_flow": self.metric_effective_flow,
            "avg_queue_length": (self.metric_queue_sum / self.metric_steps
                                 if self.metric_steps else 0.0),
        }

    def sample(self):
        metrics = get_lane_metrics(self.metric_lanes)
        self.metric_waiting_time += metrics["halting"]
        self.metric_effective_flow += metrics["moving"] * self.metric_delta_t
        self.metric_queue_sum += metrics["avg_queue"]
        self.metric_steps += 1
        now = traci.simulation.getTime()
        self.secondary_wait += sum(traci.edge.getLastStepHaltingNumber(edge)
                                   for edge in self.secondary_edges)
        J2_phase = traci.trafficlight.getPhase("J2")
        self.J2_phase_counts[J2_phase] += 1
        if J2_phase == self.last_J2_phase:
            self.J2_phase_ticks += 1
        else:
            self.J2_phase_runs.append({"phase": self.last_J2_phase,
                                       "duration": self.J2_phase_ticks,
                                       "ended_at": int(now)})
            self.last_J2_phase = J2_phase
            self.J2_phase_ticks = 1
        for tls, starts, target in (("J10", self.J10_phase_starts, 4),
                                    ("J16", self.J16_phase_starts, 2)):
            phase = traci.trafficlight.getPhase(tls)
            if phase == target and self.last_other_phase[tls] != target:
                starts.append(now)
            self.last_other_phase[tls] = phase
        for vid in traci.vehicle.getIDList():
            if vid not in self.broad_cohort and vid not in self.cohort:
                continue
            edge = traci.vehicle.getRoadID(vid)
            speed = traci.vehicle.getSpeed(vid)
            rec = self.records.setdefault(vid, {"E1_start": None, "E5_start": None,
                                                 "stopped_E1": False,
                                                 "E1_wait": 0, "first_stop": None})
            if edge == "E1":
                if rec["E1_start"] is None:
                    rec["E1_start"] = now
                    if self.active_release is not None and vid in self.cohort:
                        self.active_release["E1_entries"].append({"vehicle_id": vid,
                                                                  "time": int(now)})
                if speed < 0.1:
                    rec["stopped_E1"] = True
                    rec["E1_wait"] += 1
                    if rec["first_stop"] is None:
                        rec["first_stop"] = now
            elif edge == "E5" and rec["E5_start"] is None:
                rec["E5_start"] = now


class ObservedRun(Observer):
    def __init__(self, route):
        super().__init__(END, route)
        self.audit = FlowAudit(route)
        self.network_wait = 0
        self.tls_wait = {tls: 0 for tls in TLS}
        self.tls_edges = {
            tls: sorted({lane.rsplit("_", 1)[0] for lane in
                         traci.trafficlight.getControlledLanes(tls)})
            for tls in TLS}
        self.departed = set()
        self.arrived = set()
        self.teleports = set()
        self.speed_subscribed = set()
        self.speed_subscription_checks = 0
        self.timeline = []

    def begin_release(self, start, green):
        super().begin_release(start, green)
        self.audit.begin_release(self)

    def sample(self):
        super().sample()
        self.audit.sample()
        for vid in traci.vehicle.getIDList():
            if vid not in self.speed_subscribed:
                traci.vehicle.subscribe(vid, [tc.VAR_SPEED])
                self.speed_subscribed.add(vid)
            speed = traci.vehicle.getSubscriptionResults(vid)[tc.VAR_SPEED]
            if int(traci.simulation.getTime()) % 120 == 0:
                if speed != traci.vehicle.getSpeed(vid):
                    raise RuntimeError("Speed subscription differs from direct telemetry")
                self.speed_subscription_checks += 1
            self.network_wait += speed < 0.1
        for tls, edges in self.tls_edges.items():
            self.tls_wait[tls] += sum(traci.edge.getLastStepHaltingNumber(edge)
                                      for edge in edges)
        self.departed.update(traci.simulation.getDepartedIDList())
        self.arrived.update(traci.simulation.getArrivedIDList())
        self.teleports.update(traci.simulation.getStartingTeleportIDList())
        self.timeline.append({"time": int(traci.simulation.getTime()),
                              "J2_phase": traci.trafficlight.getPhase("J2"),
                              "J2_spent": traci.trafficlight.getSpentDuration("J2"),
                              "J2_next_switch": traci.trafficlight.getNextSwitch("J2")})


class CorridorObserver(ObservedRun):
    def __init__(self, route):
        super().__init__(route)
        prefix = ["-E0", "E1"]
        self.corridor_cohorts = {}
        for tls, (_, incoming, outgoing) in RECEIVERS.items():
            prefix.append(outgoing)
            self.corridor_cohorts[tls] = route_ids(route, prefix, END - 180)
        self.corridor_records = {vid: {} for cohort in self.corridor_cohorts.values() for vid in cohort}

    def sample(self):
        super().sample()
        now = int(traci.simulation.getTime())
        for vid in traci.vehicle.getIDList():
            if vid not in self.corridor_records:
                continue
            road = traci.vehicle.getRoadID(vid)
            speed = traci.vehicle.getSpeed(vid)
            for tls, (_, incoming, outgoing) in RECEIVERS.items():
                if vid not in self.corridor_cohorts[tls]:
                    continue
                row = self.corridor_records[vid].setdefault(tls, {
                    "entered_at": None, "crossed_at": None, "stopped": False, "waiting_s": 0})
                if road == incoming:
                    if row["entered_at"] is None:
                        row["entered_at"] = now
                    if speed < 0.1:
                        row["stopped"] = True
                        row["waiting_s"] += 1
                elif row["entered_at"] is not None and row["crossed_at"] is None:
                    if road.startswith(f":{tls}_") or road == outgoing:
                        row["crossed_at"] = now


class FullObserver(CorridorObserver):
    """Passive trajectory, arrival, secondary waiting and phase measurements."""

    def __init__(self, route):
        super().__init__(route)
        self.full_exits = {}
        for vehicle in ET.parse(route).iter("vehicle"):
            if float(vehicle.get("depart")) > END - 180:
                continue
            edges = vehicle.find("route").get("edges").split()
            for i in range(len(edges) - 4):
                if edges[i:i + 4] == ["-E0", "E1", "E5", "E10"]:
                    self.full_exits[vehicle.get("id")] = edges[i + 4]
        self.all_ids = set(self.full_exits) | set().union(*self.corridor_cohorts.values())
        self.trajectories = {vid: {tls: {
            "entered_at": None, "crossed_at": None, "arrival_at": None,
            "signal_at_arrival": None, "arrival_basis": None,
            "stopped": False, "waiting_s": 0, "stops": 0, "last_stopped": False}
            for tls in RECEIVERS} for vid in self.all_ids}
        self.secondary = {tls: 0 for tls in RECEIVERS}
        self.runs = {tls: [] for tls in RECEIVERS}
        self.phase = {tls: traci.trafficlight.getPhase(tls) for tls in RECEIVERS}
        self.started = {tls: 1 for tls in RECEIVERS}
        self.initial_partial = {tls: (
            traci.trafficlight.getSpentDuration(tls) > 0 or
            round(traci.trafficlight.getNextSwitch(tls)) !=
            traci.trafficlight.getAllProgramLogics(tls)[0].phases[self.phase[tls]].duration)
            for tls in RECEIVERS}
        self.full_timeline = []

    def sample(self):
        super().sample()
        now = int(traci.simulation.getTime())
        snapshot = {"time": now}
        for tls, (_, incoming, _) in RECEIVERS.items():
            phase = traci.trafficlight.getPhase(tls)
            snapshot[f"{tls}_phase"] = phase
            self.secondary[tls] += sum(traci.edge.getLastStepHaltingNumber(e)
                                       for e in self.tls_edges[tls] if e != incoming)
            if phase != self.phase[tls]:
                self.runs[tls].append({"tls": tls, "phase": self.phase[tls],
                    "start": self.started[tls], "ended_at": now,
                    "duration": now - self.started[tls],
                    "partial": not self.runs[tls] and self.initial_partial[tls]})
                self.phase[tls], self.started[tls] = phase, now
        self.full_timeline.append(snapshot)
        for vid in traci.vehicle.getIDList():
            if vid not in self.all_ids:
                continue
            road = traci.vehicle.getRoadID(vid)
            speed = traci.vehicle.getSpeed(vid)
            next_tls = {item[0]: item for item in traci.vehicle.getNextTLS(vid)}
            for tls, (_, incoming, outgoing) in RECEIVERS.items():
                outgoing = self.full_exits.get(vid, outgoing) if tls == "J16" else outgoing
                row = self.trajectories[vid][tls]
                if road == incoming:
                    if row["entered_at"] is None:
                        row["entered_at"] = now
                    stopped = speed < 0.1
                    row["waiting_s"] += stopped
                    row["stops"] += stopped and not row["last_stopped"]
                    row["stopped"] |= stopped
                    row["last_stopped"] = stopped
                    item = next_tls.get(tls)
                    if row["arrival_at"] is None and item and item[2] <= 5:
                        row.update(arrival_at=now, arrival_basis="first_within_5m",
                                   signal_at_arrival=traci.trafficlight.getRedYellowGreenState(tls)[item[1]])
                else:
                    row["last_stopped"] = False
                    if row["entered_at"] is not None and row["crossed_at"] is None:
                        if road.startswith(f":{tls}_") or road == outgoing:
                            row["crossed_at"] = now
                            if row["arrival_at"] is None:
                                # A crossing between 1 s samples is an upper bound;
                                # the signal before that crossing cannot be recovered.
                                row.update(arrival_at=now, arrival_basis="crossing_upper_bound",
                                           signal_at_arrival="transition_unknown")

    def finish(self):
        super().finish()
        end = int(traci.simulation.getTime()) + 1
        for tls in RECEIVERS:
            self.runs[tls].append({"tls": tls, "phase": self.phase[tls],
                "start": self.started[tls], "ended_at": end,
                "duration": end - self.started[tls], "partial": True})


def group_summary(obs, ids, last_tls="J16"):
    tls_list = list(RECEIVERS)[:list(RECEIVERS).index(last_tls) + 1]
    complete = [vid for vid in ids if obs.trajectories[vid][last_tls]["crossed_at"] is not None]
    no_stop = sum(not any(obs.trajectories[vid][t]["stopped"] for t in tls_list) for vid in complete)
    return {"cohort": len(ids), "completed": len(complete), "pending": len(ids) - len(complete),
        "no_stop_all_three": no_stop, "no_stop_all_three_pct": 100 * no_stop / len(ids) if ids else None,
        "stops_per_vehicle": statistics.mean(sum(obs.trajectories[v][t]["stops"] for t in tls_list)
                                               for v in complete) if complete else None,
        "waiting_total_s": sum(obs.trajectories[v][t]["waiting_s"] for v in ids for t in tls_list),
        "J0_to_J16_s": distribution(
            obs.trajectories[v][last_tls]["crossed_at"] - obs.audit.vehicles[v]["J0_cross_time"]
            for v in complete),
        "no_stop_per_tls": {t: sum(not obs.trajectories[v][t]["stopped"] for v in complete)
                            for t in tls_list}}


def add_audit(summary, obs):
    summary["full_corridor_any_exit"] = group_summary(obs, list(obs.full_exits))
    summary["full_corridor_straight_exit"] = group_summary(obs,
        [v for v, edge in obs.full_exits.items() if edge == "E13"])
    summary["secondary_wait_s"] = obs.secondary
    summary["controlled_secondary_wait_s"] = sum(obs.secondary.values())
    vehicle_rows = []
    for vid in sorted(obs.all_ids):
        row = {"vehicle_id": vid, "exit_J16": obs.full_exits.get(vid),
               "J0_cross_time": obs.audit.vehicles.get(vid, {}).get("J0_cross_time")}
        for tls, details in obs.trajectories[vid].items():
            row.update({f"{tls}_{k}": v for k, v in details.items() if k != "last_stopped"})
        vehicle_rows.append(row)
    summary["constraint_violations"] = validate_phases(obs.runs)
    return vehicle_rows


def validate_phases(runs):
    for tls, rows in runs.items():
        for r in rows:
            if r["partial"]:
                continue
            if (r["phase"] % 2 and r["duration"] != 4) or (
                    not r["phase"] % 2 and not 5 <= r["duration"] <= 45):
                raise RuntimeError(f"{tls}: illegal duration {r}")
        if any(b["phase"] != (a["phase"] + 1) % 8 for a, b in zip(rows, rows[1:])):
            raise RuntimeError(f"{tls}: illegal phase order")
    return 0


def summarize_copy(seed, route, obs, controllers, actions, j0_runs, programs, metadata, travel_times, delays):
    strict = {vid: r for vid, r in obs.records.items()
              if vid in obs.cohort and r["E5_start"] is not None}
    summary = {**obs.lane_metrics(), "mode": "v2", "variant": "copy_green",
               "seed": seed, "SUMO_seed": 42, "horizon_s": END,
               "travel_times_s": list(travel_times), "cumulative_delays_s": delays,
               "offsets": list(OFFSETS), "runtime_programs": programs,
               "checkpoint": str(CHECKPOINT), "checkpoint_metadata": metadata,
               "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
               "network_sha256": hashlib.sha256(NET.read_bytes()).hexdigest(),
               "route_sha256": hashlib.sha256(route.read_bytes()).hexdigest(),
               "cohort": len(obs.cohort), "crossed": len(strict),
               "no_stop": sum(not r["stopped_E1"] for r in strict.values()),
               "no_stop_pct": 100 * sum(not r["stopped_E1"] for r in strict.values()) / len(obs.cohort),
               "waiting_E1_s": sum(r["E1_wait"] for r in strict.values()),
               "mean_E1_to_E5_s": statistics.mean(r["E5_start"] - r["E1_start"] for r in strict.values()),
               "secondary_wait_J2_s": obs.secondary_wait,
               "network_total_wait_s": obs.network_wait, "tls_wait_s": obs.tls_wait,
               "pending": {"min_expected": traci.simulation.getMinExpectedNumber(),
                           "active": len(traci.vehicle.getIDList()),
                           "pending_insertion": len(traci.simulation.getPendingVehicles())},
               "arrived": len(obs.arrived), "departed": len(obs.departed),
               "teleported": len(obs.teleports), "constraint_violations": 0,
               "receivers": {}}
    events, phase_rows = [], []
    for tls, controller in controllers.items():
        copied = [e for e in controller.events if e["status"] == "copied"]
        summary["receivers"][tls] = {
            "events": len(controller.events), "completed_copies": len(copied),
            "exact_duration_copies": sum(e["receiver_green_executed"] == e["J0_green"] for e in copied),
            "start_targets_reachable": sum(e.get("start_target_reachable", False) for e in controller.events),
            "start_errors_s": distribution(e["target_error"] for e in controller.events),
            "mean_window_coverage_pct": statistics.mean(e["window_coverage_pct"] for e in controller.events),
            "unserved_at_horizon": sum(e["actual_start"] is None for e in controller.events),
            "terminal_censored_copies": sum(e.get("copy_censored", False) for e in controller.events),
        }
        events.extend(controller.events)
        phase_rows.extend(dict(r, tls=tls) for r in controller.phase_runs + [controller.partial_run])
    phase_rows.extend(dict(r, tls="J0") for r in j0_runs)
    for tls, cohort in obs.corridor_cohorts.items():
        records = [obs.corridor_records[vid].get(tls, {}) for vid in cohort]
        crossed = [r for r in records if r.get("crossed_at") is not None]
        summary["receivers"][tls].update(
            cohort=len(cohort), crossed=len(crossed),
            no_stop=sum(not r["stopped"] for r in crossed),
            no_stop_pct=100 * sum(not r["stopped"] for r in crossed) / len(cohort) if cohort else None,
            waiting_s=sum(r.get("waiting_s", 0) for r in records))
    full = obs.corridor_cohorts["J16"]
    completed = [vid for vid in full if obs.corridor_records[vid].get("J16", {}).get("crossed_at") is not None]
    summary["full_corridor"] = {
        "cohort": len(full), "completed": len(completed),
        "no_stop_all_three": sum(not any(obs.corridor_records[vid][tls]["stopped"]
                                          for tls in RECEIVERS) for vid in completed),
        "J0_to_J16_s": distribution(
            obs.corridor_records[vid]["J16"]["crossed_at"] - obs.audit.vehicles[vid]["J0_cross_time"]
            for vid in completed if obs.audit.vehicles[vid]["J0_cross_time"] is not None)}
    vehicle_rows = []
    for vid, row in obs.audit.vehicles.items():
        if row["J0_cross_time"] is None:
            continue
        saved = {k: v for k, v in row.items() if not k.startswith("last_E1_")}
        for tls, details in obs.corridor_records.get(vid, {}).items():
            saved.update({f"{tls}_{k}": v for k, v in details.items()})
        vehicle_rows.append(saved)
    per_vehicle = add_audit(summary, obs)
    summary["coordination_scope"] = "full"
    return summary, obs.release_windows, vehicle_rows, events, obs.timeline, actions, phase_rows, per_vehicle


def summarize_window(seed, mode, route, obs, controllers, actions, j0_runs, programs, delays):
    causal_window = mode == "v3" or mode.startswith("v3.")
    strict = {v: r for v, r in obs.records.items() if v in obs.cohort and r["E5_start"] is not None}
    summary = {**obs.lane_metrics(), "seed": seed, "mode": mode, "coordination_scope": "full",
        "horizon_s": END, "SUMO_seed": 42, "cumulative_delays_s": delays,
        "offsets": list(OFFSETS), "runtime_programs": programs,
        "network_sha256": hashlib.sha256(NET.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "route_sha256": hashlib.sha256(route.read_bytes()).hexdigest(),
        "cohort": len(obs.cohort), "crossed": len(strict),
        "no_stop": sum(not r["stopped_E1"] for r in strict.values()),
        "no_stop_pct": 100 * sum(not r["stopped_E1"] for r in strict.values()) / len(obs.cohort),
        "waiting_E1_s": sum(r["E1_wait"] for r in strict.values()),
        "mean_E1_to_E5_s": statistics.mean(r["E5_start"] - r["E1_start"] for r in strict.values()),
        "secondary_wait_J2_s": obs.secondary_wait, "network_total_wait_s": obs.network_wait,
        "tls_wait_s": obs.tls_wait, "arrived": len(obs.arrived), "departed": len(obs.departed),
        "teleported": len(obs.teleports), "pending": {
            "min_expected": traci.simulation.getMinExpectedNumber(),
            "active": len(traci.vehicle.getIDList()),
            "pending_insertion": len(traci.simulation.getPendingVehicles())}, "receivers": {}}
    per_vehicle = add_audit(summary, obs)
    releases, events = [], []
    timeline = {r["time"]: r for r in obs.full_timeline}
    for tls, ctrl in controllers.items():
        greens = [(r["start"], r["ended_at"]) for r in obs.runs[tls]
                  if r["phase"] == ctrl.corridor_phase]
        valid = []
        for rel in obs.release_windows:
            event = ctrl.events[rel["id"]]
            members = [c for c in obs.audit.crossings if c["release_id"] == rel["id"]]
            front = min(c["time"] for c in members) + delays[tls] if members else None
            tail = max(c["time"] for c in members) + delays[tls] if members else None
            row = {"tls": tls, "release_id": rel["id"], "J0_start": rel["start"],
                   "J0_green_s": rel["end"] - rel["start"], "vehicles": len(members),
                   "front": front, "tail": tail, "actual_opening": event.get("actual_start"),
                   "event_status": event["status"], "censored": bool(members and tail >= END)}
            if members:
                row.update(window_s=tail - front + 1,
                    covered_s=overlap(front, tail + 1, greens),
                    front_green=any(a <= front < b for a, b in greens),
                    tail_green=any(a <= tail < b for a, b in greens))
            if causal_window:
                if event["actual_J0_start"] != rel["start"] or event["first_cross_updates"] != bool(members):
                    raise RuntimeError(f"{tls}: duplicated/inconsistent V3 event")
                if members and (event["front_target"] != front or event["tail_target"] != tail):
                    raise RuntimeError(f"{tls}: noncausal window")
                if members and tail < END and event.get("front_reachable_at_first_cross") and not row["front_green"]:
                    raise RuntimeError(f"{tls}: reachable front not green: {event['id']}")
            releases.append(row)
            if members and not row["censored"]:
                valid.append(row)
        for t, item in timeline.items():
            if (item[f"{tls}_phase"] == ctrl.corridor_phase) != any(a <= t < b for a, b in greens):
                raise RuntimeError(f"{tls}: sampled phase/interval mismatch")
        noticed = [e for e in ctrl.events if e.get("front_reachable_at_announcement") is not None]
        reach = [e for e in noticed if e["front_reachable_at_announcement"]]
        summary["receivers"][tls] = {
            "events": len(ctrl.events), "eligible_notices": len(noticed),
            "reachable_at_notice": len(reach),
            "empty_preparations": sum(not e["crossings"] for e in noticed) if causal_window else None,
            "actual_openings": sum(e.get("actual_start") is not None for e in ctrl.events),
            "window_coverage_pct": 100 * sum(r["covered_s"] for r in valid) /
                sum(r["window_s"] for r in valid) if valid else None,
            "front_green": sum(r["front_green"] for r in valid), "occupied_windows": len(valid),
            "corridor_green_s": distribution(r["duration"] for r in obs.runs[tls]
                if r["phase"] == ctrl.corridor_phase and not r["partial"]),
            "opening_error_s": distribution(e.get("target_error") for e in ctrl.events),
            "unserved_at_horizon": sum(e.get("actual_start") is None and
                (not causal_window or bool(e["crossings"])) for e in ctrl.events)}
        if causal_window and (ctrl.next_crossing != len(obs.audit.crossings) or
                            not all(e["closed"] for e in ctrl.events)):
            raise RuntimeError(f"{tls}: unconsumed crossing/open request")
        events.extend(dict(e, tls=tls) for e in ctrl.events)
    phase_rows = j0_runs + [r for rows in obs.runs.values() for r in rows]
    return summary, releases, list(obs.audit.vehicles.values()), events, obs.full_timeline, actions, phase_rows, per_vehicle
