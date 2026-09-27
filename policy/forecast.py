"""Read-only corridor forecasts for deterministic SUMO shadow runs.

Signed error is predicted ETA minus real stop-line crossing time: positive is a
late prediction, negative is an early prediction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev


CORRIDOR = (("J2", "E1"), ("J10", "E5"), ("J16", "E10"))
MASTER_INCOMING = frozenset(("-E0", "-E3", "E2"))
SLAVES = tuple(slave for slave, _ in CORRIDOR)


def classify_route(route: tuple[str, ...], e1_index: int) -> tuple[str, dict[str, bool]]:
    """Classify contiguous northbound travel and stop-line applicability."""
    tail = route[e1_index:]
    if not tail or tail[0] != "E1":
        raise ValueError("E1 must be the current route edge")
    to_e5 = len(tail) > 1 and tail[1] == "E5"
    to_e10 = to_e5 and len(tail) > 2 and tail[2] == "E10"
    route_class = "e1_e5_e10" if to_e10 else "e1_e5" if to_e5 else "e1_only"
    return route_class, {
        "J2": len(tail) > 1,
        "J10": to_e5 and len(tail) > 2,
        "J16": to_e10 and len(tail) > 3,
    }


def baseline_eta(now: float, distance: float, expected_speed: float) -> float | None:
    if not (math.isfinite(now) and math.isfinite(distance) and math.isfinite(expected_speed)):
        return None
    if distance < 0 or expected_speed <= 0:
        return None
    return now + distance / expected_speed


@dataclass
class TrackedVehicle:
    vehicle_id: str
    detected_at: float
    source_edge: str
    route: tuple[str, ...]
    route_class: str
    platoon_id: int
    lane_position: float
    eta_by_slave: dict[str, float | None]
    distance_by_slave: dict[str, float | None]
    status_by_slave: dict[str, str]
    actual_by_slave: dict[str, float] = field(default_factory=dict)
    censor_reason: str | None = None


@dataclass
class Platoon:
    platoon_id: int
    first_time: float
    last_time: float
    member_ids: list[str] = field(default_factory=list)
    closed: bool = False

    def accepts(self, timestamp: float, gap: float = 3.0, window: float = 15.0) -> bool:
        return timestamp - self.last_time <= gap and timestamp - self.first_time <= window


@dataclass
class SlaveForecast:
    slave_id: str
    platoon_id: int
    revision: int
    predicted_arrival: float | None
    predicted_window_start: float | None
    predicted_window_end: float | None
    expected_vehicle_count: int
    dispersion: float | None
    quality: str
    actual_arrival: float | None = None
    prediction_error: float | None = None


class ForecastStore:
    """Latest per-slave forecasts, plus the event stream used for audit."""

    def __init__(self) -> None:
        self.by_slave: dict[str, dict[int, SlaveForecast]] = {slave: {} for slave in SLAVES}
        self.events: list[dict] = []

    def emit(self, event: str, time: float, **data: object) -> None:
        self.events.append({"event": event, "time": time, **data})

    def update(self, platoon: Platoon, vehicles: dict[str, TrackedVehicle], time: float) -> None:
        if len(platoon.member_ids) < 2:
            return
        for slave in SLAVES:
            members = [vehicles[vid] for vid in platoon.member_ids
                       if vehicles[vid].status_by_slave[slave] != "no_aplicable"]
            if not members:
                continue
            values = [vehicle.eta_by_slave[slave] for vehicle in members
                      if vehicle.eta_by_slave[slave] is not None]
            quality = "completa" if len(values) == len(members) else "parcial" if values else "no_disponible"
            previous = self.by_slave[slave].get(platoon.platoon_id)
            forecast = SlaveForecast(
                slave_id=slave,
                platoon_id=platoon.platoon_id,
                revision=1 if previous is None else previous.revision + 1,
                predicted_arrival=mean(values) if values else None,
                predicted_window_start=min(values) if values else None,
                predicted_window_end=max(values) if values else None,
                expected_vehicle_count=len(members),
                dispersion=pstdev(values) if values else None,
                quality=quality,
            )
            self.by_slave[slave][platoon.platoon_id] = forecast
            self.emit("forecast_updated", time, **vars(forecast))

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for event in self.events:
                stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


class ShadowForecast:
    """TraCI observer. All TraCI calls here are reads."""

    def __init__(self) -> None:
        self.store = ForecastStore()
        self.seen_e1: set[str] = set()
        self.followed: dict[str, TrackedVehicle] = {}
        self.previous_roads: dict[str, str] = {}
        self.platoons: dict[int, Platoon] = {}
        self.open_platoon: Platoon | None = None
        self.next_platoon_id = 1
        self.direct_e1: set[str] = set()
        self.unknown_e1: set[str] = set()
        self.teleported_e1: set[str] = set()

    @staticmethod
    def _distance_and_speed(traci, vehicle_id: str, target_edge: str) -> tuple[float | None, float | None]:
        try:
            length = traci.lane.getLength(f"{target_edge}_0")
            lane_index = min(traci.vehicle.getLaneIndex(vehicle_id), traci.edge.getLaneNumber(target_edge) - 1)
            distance = float(traci.vehicle.getDrivingDistance(
                vehicle_id, target_edge, length, laneIndex=lane_index
            ))
            edge_order = ("E1", "E5", "E10")
            speed_limits = [float(traci.lane.getMaxSpeed(f"{edge}_0"))
                            for edge in edge_order[:edge_order.index(target_edge) + 1]]
            speed = min(float(traci.vehicle.getAllowedSpeed(vehicle_id)), *speed_limits)
            if not math.isfinite(distance) or distance < 0:
                return None, speed
            return distance, speed
        except (ValueError, RuntimeError, traci.TraCIException):
            return None, None

    def _close_open(self, time: float) -> None:
        platoon = self.open_platoon
        if platoon is None:
            return
        platoon.closed = True
        self.store.emit("platoon_closed" if len(platoon.member_ids) >= 2 else "singleton_closed",
                        time, platoon_id=platoon.platoon_id, member_ids=list(platoon.member_ids))
        self.open_platoon = None

    def _add_to_platoon(self, vehicle_id: str, time: float) -> int:
        if self.open_platoon is not None and not self.open_platoon.accepts(time):
            self._close_open(time)
        if self.open_platoon is None:
            platoon = Platoon(self.next_platoon_id, time, time)
            self.next_platoon_id += 1
            self.platoons[platoon.platoon_id] = platoon
            self.open_platoon = platoon
        self.open_platoon.member_ids.append(vehicle_id)
        self.open_platoon.last_time = time
        return self.open_platoon.platoon_id

    def _detect(self, traci, vehicle_id: str, time: float, departed: set[str], teleported: set[str]) -> None:
        self.seen_e1.add(vehicle_id)
        route = tuple(traci.vehicle.getRoute(vehicle_id))
        route_index = int(traci.vehicle.getRouteIndex(vehicle_id))
        previous = self.previous_roads.get(vehicle_id)
        if vehicle_id in teleported:
            self.teleported_e1.add(vehicle_id)
            self.store.emit("e1_excluded", time, vehicle_id=vehicle_id, reason="teleport")
            return
        if route_index == 0 and vehicle_id in departed:
            self.direct_e1.add(vehicle_id)
            self.store.emit("e1_excluded", time, vehicle_id=vehicle_id, reason="direct_departure")
            return
        if (route_index <= 0 or route_index >= len(route) or route[route_index] != "E1"
                or route[route_index - 1] not in MASTER_INCOMING
                or previous not in MASTER_INCOMING and not (previous or "").startswith(":J0_")):
            self.unknown_e1.add(vehicle_id)
            self.store.emit("e1_excluded", time, vehicle_id=vehicle_id, reason="unverified_master_crossing",
                            previous_road=previous)
            return
        route_class, applicable = classify_route(route, route_index)
        platoon_id = self._add_to_platoon(vehicle_id, time)
        eta_by_slave: dict[str, float | None] = {}
        distance_by_slave: dict[str, float | None] = {}
        status_by_slave: dict[str, str] = {}
        for slave, edge in CORRIDOR:
            if not applicable[slave]:
                status_by_slave[slave] = "no_aplicable"
                eta_by_slave[slave] = None
                distance_by_slave[slave] = None
                continue
            distance, speed = self._distance_and_speed(traci, vehicle_id, edge)
            eta = baseline_eta(time, distance, speed) if distance is not None and speed is not None else None
            status_by_slave[slave] = "pendiente"
            eta_by_slave[slave] = eta
            distance_by_slave[slave] = distance
        tracked = TrackedVehicle(
            vehicle_id=vehicle_id, detected_at=time, source_edge=route[route_index - 1],
            route=route, route_class=route_class, platoon_id=platoon_id,
            lane_position=float(traci.vehicle.getLanePosition(vehicle_id)),
            eta_by_slave=eta_by_slave, distance_by_slave=distance_by_slave,
            status_by_slave=status_by_slave,
        )
        self.followed[vehicle_id] = tracked
        self.store.emit("vehicle_detected", time, vehicle_id=vehicle_id, source_edge=tracked.source_edge,
                        route=list(route), route_class=route_class, platoon_id=platoon_id,
                        lane_position=tracked.lane_position, eta_by_slave=eta_by_slave,
                        distance_by_slave=distance_by_slave, status_by_slave=dict(status_by_slave))
        self.store.update(self.platoons[platoon_id], self.followed, time)

    def observe_step(self, traci) -> None:
        time = float(traci.simulation.getTime())
        ids = set(traci.vehicle.getIDList())
        roads = {vid: traci.vehicle.getRoadID(vid) for vid in sorted(ids)}
        departed = set(traci.simulation.getDepartedIDList())
        teleported = (set(traci.simulation.getStartingTeleportIDList()) |
                      set(traci.simulation.getEndingTeleportIDList()))
        for vid in sorted(ids):
            if roads[vid] == "E1" and vid not in self.seen_e1:
                self._detect(traci, vid, time, departed, teleported)
        if self.open_platoon is not None and not self.open_platoon.accepts(time):
            self._close_open(time)
        for vid, tracked in sorted(self.followed.items()):
            if (tracked.censor_reason is not None or
                    not any(status == "pendiente" for status in tracked.status_by_slave.values())):
                continue
            if vid in teleported:
                tracked.censor_reason = "teleport"
                self.store.emit("vehicle_censored", time, vehicle_id=vid, reason="teleport")
                continue
            if vid not in ids:
                if any(status == "pendiente" for status in tracked.status_by_slave.values()):
                    tracked.censor_reason = "disappeared"
                    self.store.emit("vehicle_censored", time, vehicle_id=vid, reason="disappeared")
                continue
            if tuple(traci.vehicle.getRoute(vid)) != tracked.route:
                tracked.censor_reason = "route_changed"
                self.store.emit("vehicle_censored", time, vehicle_id=vid, reason="route_changed")
                continue
            previous = self.previous_roads.get(vid)
            current = roads[vid]
            for slave, edge in CORRIDOR:
                if tracked.status_by_slave[slave] != "pendiente" or previous != edge or current == edge:
                    continue
                if current.startswith(f":{slave}_") or current in tracked.route:
                    tracked.status_by_slave[slave] = "cruzado"
                    tracked.actual_by_slave[slave] = time
                    predicted = tracked.eta_by_slave[slave]
                    self.store.emit("actual_arrival", time, vehicle_id=vid, slave_id=slave,
                                    platoon_id=tracked.platoon_id, predicted_arrival=predicted,
                                    error_firmado=predicted - time if predicted is not None else None,
                                    error_absoluto=abs(predicted - time) if predicted is not None else None)
        self.store.emit("slave_signal", time, signals={
            slave: {
                "program": traci.trafficlight.getProgram(slave),
                "phase": traci.trafficlight.getPhase(slave),
                "state": traci.trafficlight.getRedYellowGreenState(slave),
                "next_switch": traci.trafficlight.getNextSwitch(slave),
            } for slave in SLAVES
        })
        self.previous_roads = roads

    def finalize(self, time: float) -> dict:
        self._close_open(time)
        summary = {}
        for slave in SLAVES:
            members = [vehicle for vehicle in self.followed.values()
                       if vehicle.status_by_slave[slave] != "no_aplicable"]
            valid = [vehicle for vehicle in members if vehicle.censor_reason is None]
            observed = [vehicle for vehicle in valid if slave in vehicle.actual_by_slave]
            scored = [vehicle for vehicle in observed if vehicle.eta_by_slave[slave] is not None]
            errors = [vehicle.eta_by_slave[slave] - vehicle.actual_by_slave[slave] for vehicle in scored]
            group_errors = []
            for platoon_id, forecast in self.store.by_slave[slave].items():
                platoon = self.platoons[platoon_id]
                group = [self.followed[vid] for vid in platoon.member_ids
                         if self.followed[vid].status_by_slave[slave] != "no_aplicable"]
                if (group and all(vehicle.censor_reason is None and slave in vehicle.actual_by_slave
                                  and vehicle.eta_by_slave[slave] is not None for vehicle in group)):
                    forecast.actual_arrival = mean(vehicle.actual_by_slave[slave] for vehicle in group)
                    forecast.prediction_error = forecast.predicted_arrival - forecast.actual_arrival
                    group_errors.append(forecast.prediction_error)
                    self.store.emit("platoon_result", time, slave_id=slave, platoon_id=platoon_id,
                                    actual_arrival=forecast.actual_arrival,
                                    error_firmado=forecast.prediction_error,
                                    error_absoluto=abs(forecast.prediction_error))
            summary[slave] = {
                "applicable": len(members),
                "no_aplicable": len(self.followed) - len(members),
                "forecast_valid": sum(vehicle.eta_by_slave[slave] is not None for vehicle in members),
                "observed_crossings": len(observed),
                "coverage": len(observed) / len(members) if members else None,
                "mae": mean(abs(error) for error in errors) if errors else None,
                "bias": mean(errors) if errors else None,
                "scored_vehicles": len(errors),
                "platoon_mae": mean(abs(error) for error in group_errors) if group_errors else None,
                "platoon_bias": mean(group_errors) if group_errors else None,
                "scored_platoons": len(group_errors),
                "route_changed": sum(vehicle.censor_reason == "route_changed" for vehicle in members),
                "teleport": sum(vehicle.censor_reason == "teleport" for vehicle in members),
                "disappeared": sum(vehicle.censor_reason == "disappeared" for vehicle in members),
                "horizon_pending": sum(vehicle.censor_reason is None and
                                       vehicle.status_by_slave[slave] == "pendiente" for vehicle in members),
            }
        result = {
            "error_sign_convention": "predicted_minus_actual; positive=late; negative=early",
            "detected_master": len(self.followed),
            "direct_e1": len(self.direct_e1),
            "unknown_e1": len(self.unknown_e1),
            "teleported_e1": len(self.teleported_e1),
            "platoons": sum(len(p.member_ids) >= 2 for p in self.platoons.values()),
            "singletons": sum(len(p.member_ids) == 1 for p in self.platoons.values()),
            "slaves": summary,
        }
        self.store.emit("summary", time, **result)
        return result
