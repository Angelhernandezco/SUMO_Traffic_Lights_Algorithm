"""Pre-C2 observation only. Imported by the evaluation harness, never PPO/C1.

Times are first observed samples (one existing SUMO step resolution). Signal
colours describe the sampled state, not a proven cause of a stopped vehicle.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import pstdev

from policy.forecast import CORRIDOR

ACCESS = dict(CORRIDOR)
TARGET = {edge: tls for tls, edge in CORRIDOR}
NEXT = {"J2": "J10", "J10": "J16"}


class ReadOnlyDomain:
    def __init__(self, domain):
        self._domain = domain

    def __getattr__(self, name):
        if not name.startswith("get"):
            raise AssertionError(f"Telemetry cannot invoke {name}")
        return getattr(self._domain, name)


class ReadOnlyTraCI:
    def __init__(self, api):
        self.TraCIException = api.TraCIException
        for name in ("vehicle", "lane", "edge", "simulation", "trafficlight"):
            setattr(self, name, ReadOnlyDomain(getattr(api, name)))

    def __getattr__(self, name):
        raise AssertionError(f"Telemetry cannot invoke {name}")


def temporal_spread(times):
    values = sorted(times)
    return {"observed_members": len(values),
            "head_tail_seconds": values[-1] - values[0] if values else None,
            "std_seconds": pstdev(values) if values else None,
            "maximum_gap_seconds": max((b - a for a, b in zip(values, values[1:])), default=0) if values else None,
            "compact_observed_subset": (values[-1] - values[0] <= 15
                                        and all(b - a <= 3 for a, b in zip(values, values[1:]))) if len(values) >= 2 else None}


class CorridorTelemetry:
    def __init__(self):
        self.events, self.samples, self.platoon_steps = [], [], []
        self.previous, self.identities, self.visits, self.active = {}, {}, {}, {}
        self.censored = {}
        self.release = {}
        self.phase_previous, self.receptor_cycles = {}, Counter()
        self.steps, self.last_time = 0, None

    def emit(self, event, now, vid, **data):
        identity = self.identities.get(vid, {})
        self.events.append({"event": event, "time": now, "vehicle_id": vid,
                            "platoon_id": identity.get("platoon_id"), **data})

    def _censor(self, now, vid, reason):
        if vid not in self.censored:
            self.censored[vid] = reason
            self.emit("vehicle_censored", now, vid, reason=reason)
            self.active.pop(vid, None)

    def _free_flow(self, api, vid, tls):
        """Causal constant-speed estimate; acceleration/queue/signals excluded."""
        try:
            edge = ACCESS[tls]
            lane_index = min(int(api.vehicle.getLaneIndex(vid)), int(api.edge.getLaneNumber(edge)) - 1)
            lane = f"{edge}_{lane_index}"
            distance = float(api.vehicle.getDrivingDistance(vid, edge, api.lane.getLength(lane), laneIndex=lane_index))
            speed = min(float(api.vehicle.getAllowedSpeed(vid)), float(api.lane.getMaxSpeed(lane)))
            return distance / speed if distance >= 0 and speed > 0 and math.isfinite(distance) else None
        except (ValueError, RuntimeError, api.TraCIException):
            return None

    def observe(self, api, source):
        now, dt = float(api.simulation.getTime()), float(api.simulation.getDeltaT())
        if self.last_time is not None and not math.isclose(now - self.last_time, dt):
            raise ValueError("Telemetry requires one existing simulation step per observation")
        self.steps += 1
        ids = set(api.vehicle.getIDList())
        roads = {v: api.vehicle.getRoadID(v) for v in sorted(ids)}
        teleports = set(api.simulation.getStartingTeleportIDList()) | set(api.simulation.getEndingTeleportIDList())
        arrived = set(api.simulation.getArrivedIDList())
        for vid, vehicle in sorted(source.followed.items()):
            if vid not in self.release:
                self.release[vid] = vehicle.detected_at
                self.identities[vid] = {"platoon_id": vehicle.platoon_id, "route": vehicle.route, "origin": "j0_released"}
                self.emit("j0_release_observed", vehicle.detected_at, vid,
                          free_flow_to_j2_seconds=(vehicle.eta_by_slave["J2"] - vehicle.detected_at
                                                   if vehicle.eta_by_slave["J2"] is not None else None))
            if vehicle.censor_reason is not None:
                self._censor(now, vid, vehicle.censor_reason)
        for vid in sorted(self.previous):
            if vid in teleports:
                self._censor(now, vid, "teleport")
            elif vid not in ids and vid not in arrived:
                self._censor(now, vid, "disappeared")
            elif vid in arrived:
                self.emit("route_end_observed", now, vid)
                self.active.pop(vid, None)
        signals, links = {}, {}
        for tls in ACCESS:
            phase = int(api.trafficlight.getPhase(tls))
            if phase == 2 and self.phase_previous.get(tls) != 2:
                self.receptor_cycles[tls] += 1
            self.phase_previous[tls] = phase
            signals[tls] = {"phase": phase, "state": api.trafficlight.getRedYellowGreenState(tls),
                            "next_switch": float(api.trafficlight.getNextSwitch(tls)),
                            "receptor_cycle": self.receptor_cycles[tls]}
            links[tls] = api.trafficlight.getControlledLinks(tls)
        current = {}
        for vid in sorted(ids):
            road = roads[vid]
            internal_tls = next((tls for tls in ACCESS if road.startswith(f":{tls}_")), None)
            if vid not in self.identities and road not in TARGET and internal_tls is None:
                continue
            route = tuple(api.vehicle.getRoute(vid))
            index = int(api.vehicle.getRouteIndex(vid))
            if vid not in self.identities:
                self.identities[vid] = {"platoon_id": None, "route": route,
                                        "origin": "direct_e1" if vid in source.direct_e1 else "other_upstream"}
            if route != self.identities[vid]["route"]:
                self._censor(now, vid, "route_changed")
            if vid in teleports:
                self._censor(now, vid, "teleport")
            lane = api.vehicle.getLaneID(vid)
            position, speed = float(api.vehicle.getLanePosition(vid)), float(api.vehicle.getSpeed(vid))
            tls = TARGET.get(road) or internal_tls
            outgoing = route[index + 1] if road in TARGET and 0 <= index < len(route) - 1 else None
            old = self.previous.get(vid)
            # Censored IDs stay visible but can never create valid crossings again.
            if old and vid not in self.censored:
                previous_tls = TARGET.get(old["edge"])
                if previous_tls and road != old["edge"]:
                    key = self.active.pop(vid, None)
                    if key and (internal_tls == previous_tls or road == old["outgoing"]):
                        visit = self.visits[key]
                        visit.update(crossed_at=now, crossing_cycle=signals[previous_tls]["receptor_cycle"],
                                     crossing_signal_phase=signals[previous_tls]["phase"])
                        self.emit("stop_line_cross", now, vid, tls_id=previous_tls, visit=key,
                                  receptor_cycle=visit["crossing_cycle"],
                                  free_flow_to_next_seconds=(self._free_flow(api, vid, NEXT[previous_tls])
                                                            if previous_tls in NEXT else None))
                        if road == old["outgoing"]:
                            visit["exited_at"] = now
                            self.emit("intersection_exit", now, vid, tls_id=previous_tls, edge=road,
                                      resolution="cross_and_exit_in_same_step")
                    else:
                        self._censor(now, vid, "unobserved_crossing_or_route_change")
                old_internal = next((t for t in ACCESS if old["edge"].startswith(f":{t}_")), None)
                if old_internal and internal_tls != old_internal:
                    pending = [r for r in self.visits.values() if r["vehicle_id"] == vid and r["tls_id"] == old_internal
                               and r["crossed_at"] is not None and r["exited_at"] is None]
                    if pending and road == pending[-1]["outgoing"]:
                        pending[-1]["exited_at"] = now
                        self.emit("intersection_exit", now, vid, tls_id=old_internal, edge=road,
                                  resolution="internal_to_outgoing_observed")
                    elif pending:
                        self._censor(now, vid, "unobserved_intersection_exit")
            relevant = []
            if road in TARGET:
                for link_index, connections in enumerate(links[tls]):
                    if any(inc == lane and out.rsplit("_", 1)[0] == outgoing for inc, out, _ in connections):
                        relevant.append(link_index)
            colours = [signals[tls]["state"][i] for i in relevant]
            movement_green = all(c in "Gg" for c in colours) if colours else None
            colour = colours[0] if colours and len(set(colours)) == 1 else None
            ahead = sum(float(api.vehicle.getLanePosition(other)) > position
                        for other in api.lane.getLastStepVehicleIDs(lane) if other != vid)
            downstream = sorted({out for i in relevant for inc, out, _ in links[tls][i]
                                 if inc == lane and out.rsplit("_", 1)[0] == outgoing})
            occupancy = {out: float(api.lane.getLastStepOccupancy(out)) for out in downstream}
            stopped = speed < 0.1
            queue_proxy = bool(road in TARGET and stopped and ahead > 0)
            sample = {"time": now, "dt": dt, "vehicle_id": vid, **self.identities[vid],
                      "tls_id": tls, "edge": road, "lane": lane, "position": position, "speed": speed,
                      "outgoing": outgoing, "route_index": index,
                      "distance_to_stop_line": max(0.0, float(api.lane.getLength(lane)) - position) if road in TARGET else None,
                      "signal": signals.get(tls), "signal_link_indices": relevant, "movement_signal": colour,
                      "movement_green": movement_green, "receiver_green": signals[tls]["phase"] == 2 if tls else None,
                      "vehicles_ahead_same_lane": ahead, "stopped": stopped, "queue_proxy": queue_proxy,
                      "free_flow_remaining_seconds": self._free_flow(api, vid, tls) if road in TARGET and outgoing else None,
                      "downstream_occupancy_percent": occupancy, "censor_reason": self.censored.get(vid)}
            if vid not in self.censored:
                if road in TARGET and (not old or old["edge"] != road):
                    key = f"{tls}:{vid}:{index}"
                    visit = {"tls_id": tls, "vehicle_id": vid, "platoon_id": sample["platoon_id"],
                             "route_index": index, "incoming": road, "outgoing": outgoing,
                             "entered_at": now, "left_censored": old is None and not (
                                 vid in source.followed and source.followed[vid].detected_at == now),
                             "crossed_at": None, "crossing_cycle": None, "crossing_signal_phase": None,
                             "exited_at": None, "waiting": 0., "stops": 0,
                             "stopped_red": 0., "stopped_green": 0., "stopped_yellow": 0., "stopped_unknown": 0.}
                    self.visits[key], self.active[vid] = visit, key
                    self.emit("approach_enter", now, vid, tls_id=tls, visit=key, edge=road,
                              left_censored=visit["left_censored"])
                    if old and old["edge"] != road:
                        self.emit("next_link_enter", now, vid, tls_id=tls, edge=road, previous_edge=old["edge"])
                if stopped and (not old or not old["stopped"]):
                    self.emit("stop_begin", now, vid, tls_id=tls, left_censored=old is None)
                if old and old["stopped"] and not stopped:
                    self.emit("stop_end", now, vid, tls_id=tls)
                if queue_proxy and (not old or not old["queue_proxy"]):
                    self.emit("queue_proxy_enter", now, vid, tls_id=tls, definition="speed_lt_0.1_and_ahead_same_lane")
                key = self.active.get(vid)
                if key:
                    visit = self.visits[key]
                    visit["waiting"] += dt * stopped
                    visit["stops"] += int(stopped and (not old or not old["stopped"]))
                    category = "green" if movement_green else "red" if colour in ("r", "R") else "yellow" if colour in ("y", "Y") else "unknown"
                    visit[f"stopped_{category}"] += dt * stopped
            self.samples.append(sample)
            current[vid] = sample
        for pid, platoon in sorted(source.platoons.items()):
            locations = [{"vehicle_id": v, "edge": current[v]["edge"], "lane": current[v]["lane"],
                          "position": current[v]["position"], "outgoing": current[v]["outgoing"],
                          "censor_reason": current[v]["censor_reason"]} for v in platoon.member_ids if v in current]
            if locations:
                self.platoon_steps.append({"time": now, "platoon_id": pid, "source_closed": platoon.closed,
                                           "original_members": list(platoon.member_ids),
                                           "present_members": [v["vehicle_id"] for v in locations], "locations": locations,
                                           "censored_members": [v for v in platoon.member_ids if v in self.censored],
                                           "spread_across_edges": len({v["edge"] for v in locations}) > 1,
                                           "spread_across_lanes": len({v["lane"] for v in locations}) > 1})
        self.previous, self.last_time = current, now

    def export(self, source):
        """Retrospective reporting only. No result returns to the environment."""
        visits = [{**r, "censor_reason": self.censored.get(r["vehicle_id"])} for r in self.visits.values()]
        segments = []
        for vid, released in sorted(self.release.items()):
            start, upstream = released, "J0"
            for tls in ACCESS:
                rows = [r for r in visits if r["vehicle_id"] == vid and r["tls_id"] == tls]
                if not rows:
                    break
                row = rows[0]
                end = row["crossed_at"]
                samples = [s for s in self.samples if s["vehicle_id"] == vid and start < s["time"] <= (end or self.last_time)]
                release_event = next((e for e in self.events if e["vehicle_id"] == vid and
                                      e["event"] == ("j0_release_observed" if upstream == "J0" else "stop_line_cross") and
                                      (upstream == "J0" or e["tls_id"] == upstream)), {})
                segments.append({"vehicle_id": vid, "platoon_id": row["platoon_id"], "from": upstream, "to": tls,
                                 "start": start, "arrival": row["entered_at"], "cross": end,
                                 "arrival_seconds": row["entered_at"] - start, "cross_seconds": end - start if end is not None else None,
                                 "waiting": sum(s["dt"] for s in samples if s["stopped"]),
                                 "stop_begins": sum(e["event"] == "stop_begin" and e["vehicle_id"] == vid and
                                                    start < e["time"] <= (end or self.last_time) for e in self.events),
                                 "free_flow_estimate_seconds": release_event.get("free_flow_to_j2_seconds" if upstream == "J0" else "free_flow_to_next_seconds"),
                                 "censor_reason": self.censored.get(vid), "complete": end is not None and vid not in self.censored})
                if end is None:
                    break
                start, upstream = end, tls
        groups = []
        for pid, platoon in sorted(source.platoons.items()):
            for tls in ACCESS:
                rows = [r for r in visits if r["platoon_id"] == pid and r["tls_id"] == tls]
                valid = [r for r in rows if r["censor_reason"] is None]
                expected = [v for v in platoon.member_ids if any(a == ACCESS[tls] for a in source.followed[v].route[:-1])]
                entry = temporal_spread(r["entered_at"] for r in valid if not r["left_censored"])
                crossing = temporal_spread(r["crossed_at"] for r in valid if r["crossed_at"] is not None)
                exit_spread = temporal_spread(r["exited_at"] for r in valid if r["exited_at"] is not None)
                cycles = sorted({r["crossing_cycle"] for r in valid if r["crossed_at"] is not None})
                groups.append({"platoon_id": pid, "tls_id": tls, "original_members": list(platoon.member_ids),
                               "expected_members": expected, "observed_members": [r["vehicle_id"] for r in rows],
                               "entry": entry, "crossing": crossing, "exit": exit_spread,
                               "crossing_cycles": cycles, "different_observed_cycles": len(cycles) > 1,
                               "complete_group": len(expected) >= 2 and all(any(r["vehicle_id"] == v and r["exited_at"] is not None for r in valid) for v in expected)})
        complete = [g for g in groups if g["complete_group"]]
        return {"steps": self.steps, "last_time": self.last_time, "visits": visits,
                "segments": segments, "platoons": groups, "censored": self.censored,
                "compact_exit_groups": sum(g["exit"]["compact_observed_subset"] is True for g in complete),
                "complete_group_denominator": len(complete),
                "compact_exit_percent": 100 * sum(g["exit"]["compact_observed_subset"] is True for g in complete) / len(complete) if complete else None}
