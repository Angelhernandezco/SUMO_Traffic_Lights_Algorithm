"""Read-only, movement-level evaluation. Never imported by the controller.

Waiting is sampled stopped vehicle-seconds on controlled incoming lanes,
matching the existing lane halting metric (speed < 0.1 m/s). Crossing is
observed on entry to a TLS internal lane or its planned outgoing edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean

CORRIDOR_MOVEMENTS = {"J2": ("E1", "E5"), "J10": ("E5", "E10"), "J16": ("E10", "E13")}
CORRIDOR_PREFIXES = {"J2": ("E1",), "J10": ("E1", "E5"), "J16": ("E1", "E5", "E10")}


def classify_movement(tls, route, index):
    """Use the current route index, not a future observed crossing label."""
    if index < 0 or index >= len(route):
        return "unknown", None, None
    incoming = route[index]
    outgoing = route[index + 1] if index + 1 < len(route) else None
    group = "priority" if (incoming, outgoing) == CORRIDOR_MOVEMENTS[tls] else "secondary"
    return group, incoming, outgoing


def classify_origin(tls, route, index, released_by_j0, direct_e1):
    prefix = CORRIDOR_PREFIXES[tls]
    contiguous = tuple(route[max(0, index + 1 - len(prefix)):index + 1]) == prefix
    if released_by_j0 and contiguous:
        return "j0_released"
    if direct_e1 and contiguous:
        return "direct_e1"
    return "other_upstream"


@dataclass
class Approach:
    tls: str
    vehicle_id: str
    route_index: int
    route: tuple[str, ...]
    group: str
    incoming: str | None
    outgoing: str | None
    origin: str
    first_seen: float
    first_lane: str
    platoon_id: int | None
    j0_release_time: float | None
    left_censored: bool = False
    waiting: float = 0.0
    stops: int = 0
    stopped: bool = False
    last_seen: float = 0.0
    crossed_at: float | None = None
    censor_reason: str | None = None
    remanent_seconds: float = 0.0
    remanent_waiting: float = 0.0
    stops_started_as_remanent: int = 0
    selected_platoon_interventions: list[int] = field(default_factory=list)

    @property
    def key(self):
        return f"{self.tls}:{self.vehicle_id}:{self.route_index}"


def summarize(records):
    records = list(records)
    ids = {r.vehicle_id for r in records}
    crossed = [r for r in records if r.crossed_at is not None and r.censor_reason is None]
    complete = [r for r in crossed if not r.left_censored]
    travel = [r.crossed_at - r.j0_release_time for r in complete
              if r.origin == "j0_released" and r.j0_release_time is not None]
    approach_times = [r.crossed_at - r.first_seen for r in complete]
    waiting = sum(r.waiting for r in records)
    return {
        "vehicles": len(ids), "approach_visits": len(records),
        "waiting": waiting, "mean_waiting_per_vehicle": waiting / len(ids) if ids else None,
        "stops": sum(r.stops for r in records),
        "stopped_vehicles": len({r.vehicle_id for r in records if r.stops}),
        "throughput": len({r.vehicle_id for r in crossed}), "crossings": len(crossed),
        "no_stop_crossings": sum(r.stops == 0 for r in complete),
        "complete_crossings": len(complete),
        "no_stop_crossing_percent": 100 * sum(r.stops == 0 for r in complete) / len(complete) if complete else None,
        "mean_approach_seconds": mean(approach_times) if approach_times else None,
        "mean_j0_to_crossing_seconds": mean(travel) if travel else None,
        "j0_travel_samples": len(travel),
        "unfinished_or_censored": sum(r.crossed_at is None or r.censor_reason is not None for r in records),
        "left_censored": sum(r.left_censored for r in records),
        "remanent_observed_vehicles": len({r.vehicle_id for r in records if r.remanent_seconds}),
        "remanent_waiting": sum(r.remanent_waiting for r in records),
        "remanent_seconds": sum(r.remanent_seconds for r in records),
        "stops_started_as_remanent": sum(r.stops_started_as_remanent for r in records),
    }


class FlowMetricsObserver:
    """Only TraCI getters; all outcomes stay outside the policy environment."""

    def __init__(self, contexts):
        self.lane_tls = {lane: tls for tls, context in contexts.items() if tls in CORRIDOR_MOVEMENTS
                         for lane in context.controlled_lanes}
        self.records = {}
        self.active = {}
        self.previous_roads = {}
        self.step_count = 0
        self.last_time = None
        self.events = []

    def observe(self, traci, source, local, executor):
        now = float(traci.simulation.getTime())
        step = float(traci.simulation.getDeltaT())
        if self.last_time is not None and now - self.last_time != step:
            raise ValueError("Telemetry must observe exactly one existing simulation step")
        self.step_count += 1
        ids = set(traci.vehicle.getIDList())
        roads = {vid: traci.vehicle.getRoadID(vid) for vid in sorted(ids)}
        teleported = (set(traci.simulation.getStartingTeleportIDList()) |
                      set(traci.simulation.getEndingTeleportIDList()))
        # Process physical exits before new visits. A vanished/teleported
        # vehicle is never mistaken for a crossing.
        for vid, record in sorted(list(self.active.items())):
            if vid in teleported:
                record.censor_reason = "teleport"
            elif vid not in ids:
                record.censor_reason = "disappeared_or_route_ended_on_access"
            elif roads[vid] != record.incoming:
                if roads[vid].startswith(f":{record.tls}_") or roads[vid] == record.outgoing:
                    record.crossed_at = now
                    self.events.append({"event": "flow_crossing", "time": now, "key": record.key,
                                        "group": record.group, "origin": record.origin})
                else:
                    record.censor_reason = "unobserved_crossing_or_route_change"
            else:
                continue
            del self.active[vid]
        occupants = {vid: lane for lane in sorted(self.lane_tls)
                     for vid in traci.lane.getLastStepVehicleIDs(lane)}
        for vid, lane in sorted(occupants.items()):
            tls = self.lane_tls[lane]
            route = tuple(traci.vehicle.getRoute(vid))
            index = int(traci.vehicle.getRouteIndex(vid))
            if vid not in self.active:
                group, incoming, outgoing = classify_movement(tls, route, index)
                if incoming != lane.rsplit("_", 1)[0]:
                    raise ValueError(f"Unclassifiable live route/lane: {tls}/{vid}")
                followed = source.followed.get(vid)
                origin = classify_origin(tls, route, index, followed is not None, vid in source.direct_e1)
                previous = self.previous_roads.get(vid)
                record = Approach(tls, vid, index, route, group, incoming, outgoing, origin,
                                  now, lane, followed.platoon_id if followed else None,
                                  followed.detected_at if followed else None,
                                  left_censored=bool(previous == incoming or (self.step_count == 1 and index > 0)),
                                  last_seen=now)
                if record.key in self.records:
                    raise ValueError(f"Discontinuous revisit of a measured approach: {record.key}")
                self.records[record.key] = record
                self.active[vid] = record
                self.events.append({"event": "flow_enter", "time": now, "key": record.key,
                                    "incoming": incoming, "outgoing": outgoing,
                                    "group": group, "origin": origin, "left_censored": record.left_censored})
            record = self.active[vid]
            if route != record.route:
                record.censor_reason = "route_changed"
            stopped = float(traci.vehicle.getSpeed(vid)) < 0.1
            packet = local.latest[tls].get(record.platoon_id) if record.platoon_id is not None else None
            remanent = bool(packet and packet.source_closed and packet.source_member_count >= 2
                            and len(packet.valid_member_ids) == 1 and vid in packet.valid_member_ids)
            if stopped and not record.stopped:
                record.stops += 1
                record.stops_started_as_remanent += int(remanent)
                self.events.append({"event": "flow_stop", "time": now, "key": record.key,
                                    "remanent": remanent})
            record.waiting += step * stopped
            record.remanent_waiting += step * stopped * remanent
            record.remanent_seconds += step * remanent
            record.stopped, record.last_seen = stopped, now
            if executor is not None and record.platoon_id is not None:
                results = executor.results[tls]
                if (results and results[-1]["time"] == now and results[-1]["result"] == "executed"
                        and results[-1]["platoon_id"] == record.platoon_id):
                    record.selected_platoon_interventions.append(results[-1]["receptor_occurrence"])
        for tls in CORRIDOR_MOVEMENTS:
            measured = sum(float(traci.vehicle.getSpeed(vid)) < 0.1 for vid, lane in occupants.items()
                           if self.lane_tls[lane] == tls)
            halted = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in self.lane_tls
                         if self.lane_tls[lane] == tls)
            if measured != halted:
                raise ValueError(f"Stopped waiting does not reconcile with SUMO halting count at {tls}: {measured}/{halted}")
        self.previous_roads, self.last_time = roads, now

    def export(self):
        return {"steps": self.step_count, "last_time": self.last_time,
                "records": [asdict(r) for _, r in sorted(self.records.items())]}

    def summary(self):
        records = list(self.records.values())
        result = {}
        for tls in CORRIDOR_MOVEMENTS:
            selected = [r for r in records if r.tls == tls]
            receiver = CORRIDOR_MOVEMENTS[tls][0]
            result[tls] = {
                "all": summarize(selected),
                "priority": summarize(r for r in selected if r.group == "priority"),
                "secondary": summarize(r for r in selected if r.group == "secondary"),
                "receiver_access_all": summarize(r for r in selected if r.incoming == receiver),
                "receiver_access_turns_or_terminal": summarize(r for r in selected if r.incoming == receiver and r.group != "priority"),
                "other_accesses": summarize(r for r in selected if r.incoming != receiver),
                "by_origin": {origin: summarize(r for r in selected if r.group == "priority" and r.origin == origin)
                              for origin in ("j0_released", "direct_e1", "other_upstream")},
                "receiver_j0_all_movements": summarize(r for r in selected if r.incoming == receiver and r.origin == "j0_released"),
            }
        return result
