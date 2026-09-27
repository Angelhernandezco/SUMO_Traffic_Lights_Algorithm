"""C0 observations and hypothetical reservations. No traffic-light writes.

Local readiness is a free-flow lower estimate, not a stop-line crossing ETA.
The shadow ledger never changes nominal signal times or predicts C1 outcomes.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from policy.forecast import CORRIDOR, baseline_eta


TLS_ORDER = tuple(tls for tls, _ in CORRIDOR)
REFERENCE_PAIRS = dict(zip(TLS_ORDER, (22, 9, 14)))
NOMINAL_DURATIONS = {
    "J2": (15, 4, 15, 4, 15, 24, 15, 4),
    "J10": (41, 4, 41, 4),
    "J16": (15, 4, 15, 4, 15, 4, 15, 4),
}


def validate_modes(eta_mode, coordination_mode):
    if eta_mode not in ("baseline", "local"):
        raise ValueError("eta-mode must be baseline or local")
    if coordination_mode not in ("off", "shadow"):
        raise ValueError("C0 coordination-mode must be off or shadow")
    if coordination_mode == "shadow" and eta_mode != "local":
        raise ValueError("coordination-mode shadow requires eta-mode local")


@dataclass(frozen=True)
class SlaveProgram:
    tls_id: str
    program_id: str
    durations: tuple[float, ...]
    states: tuple[str, ...]
    movements: frozenset[tuple[str, str]]
    receptor: int = 2


def read_programs(traci):
    programs = {}
    for tls, edge in CORRIDOR:
        active = traci.trafficlight.getProgram(tls)
        logic = next((p for p in traci.trafficlight.getAllProgramLogics(tls)
                      if p.programID == active), None)
        if logic is None or logic.type != 0:
            raise ValueError(f"C0 requires an active static program for {tls}")
        durations = tuple(float(p.duration) for p in logic.phases)
        states = tuple(p.state for p in logic.phases)
        if durations != NOMINAL_DURATIONS[tls]:
            raise ValueError(f"Unexpected C0 phase durations for {tls}: {durations}")
        for index, phase in enumerate(logic.phases):
            successor = tuple(phase.next)
            if successor and successor != ((index + 1) % len(states),):
                raise ValueError(f"Noncyclic phase sequence for {tls}")
            green = "G" in phase.state or "g" in phase.state
            yellow = "y" in phase.state.lower()
            if green != (index % 2 == 0) or yellow != (index % 2 == 1):
                raise ValueError(f"Unexpected green/yellow sequence for {tls}")
        movements = set()
        corridor_links = 0
        for index, links in enumerate(traci.trafficlight.getControlledLinks(tls)):
            for incoming, outgoing, _ in links:
                incoming_edge = incoming.rsplit("_", 1)[0]
                if incoming_edge != edge:
                    continue
                corridor_links += 1
                if index >= len(states[2]) or states[2][index] not in "Gg":
                    raise ValueError(f"Phase 2 does not serve corridor link {index} of {tls}")
                movements.add((incoming_edge, outgoing.rsplit("_", 1)[0]))
        if not corridor_links:
            raise ValueError(f"No corridor connections found for {tls}")
        programs[tls] = SlaveProgram(tls, active, durations, states, frozenset(movements))
    return programs


@dataclass(frozen=True)
class LocalForecast:
    tls_id: str
    platoon_id: int
    revision: int
    observed_at: float
    source_closed: bool
    source_member_count: int
    member_ids: tuple[str, ...]
    valid_member_ids: tuple[str, ...]
    eligible_member_count: int
    ready_earliest: float | None
    ready_latest: float | None
    quality: str
    compatible: bool
    member_failures: tuple[tuple[str, str], ...] = ()
    estimate_kind: str = "optimistic_local_free_flow_readiness"


class LocalForecastObserver:
    """Uses source membership and live state, never arrival/error/status fields."""

    def __init__(self, programs, emit):
        self.programs, self.emit = programs, emit
        self.latest = {tls: {} for tls in TLS_ORDER}
        self.revisions = Counter()
        self.stats = {tls: Counter() for tls in TLS_ORDER}
        self.seen = {tls: set() for tls in TLS_ORDER}

    def observe(self, traci, source, now):
        current = {tls: {} for tls in TLS_ORDER}
        grouped = defaultdict(list)
        # previous_roads is the current road snapshot captured by Fase B.
        # It contains no arrival labels and is refreshed before this observer.
        for vid, road in sorted(source.previous_roads.items()):
            vehicle = source.followed.get(vid)
            if vehicle is None:
                continue
            for tls, edge in CORRIDOR:
                if road == edge:
                    grouped[(tls, vehicle.platoon_id)].append(vid)
        errors = (ValueError, RuntimeError, traci.TraCIException)
        for (tls, pid), member_ids in sorted(grouped.items()):
            platoon = source.platoons[pid]
            etas, valid_ids, failures = [], [], []
            compatible_count = 0
            eligible_count = 0
            for vid in member_ids:
                original_route = source.followed[vid].route
                movement_counted = False
                try:
                    route = tuple(traci.vehicle.getRoute(vid))
                    index = int(traci.vehicle.getRouteIndex(vid))
                    if (route != original_route or index < 0 or index + 1 >= len(route)
                            or (route[index], route[index + 1]) not in self.programs[tls].movements):
                        failures.append((vid, "incompatible"))
                        continue
                    compatible_count += 1
                    eligible_count += 1
                    movement_counted = True
                    lane = traci.vehicle.getLaneID(vid)
                    if lane.rsplit("_", 1)[0] != dict(CORRIDOR)[tls]:
                        raise ValueError("Vehicle is not on its observed corridor access")
                    distance = float(traci.lane.getLength(lane)) - float(traci.vehicle.getLanePosition(vid))
                    speed = min(float(traci.vehicle.getAllowedSpeed(vid)),
                                float(traci.lane.getMaxSpeed(lane)))
                    eta = baseline_eta(now, distance, speed)
                    if eta is None:
                        raise ValueError("Invalid local distance or speed")
                    etas.append(eta)
                    valid_ids.append(vid)
                except errors:
                    # Unknown movement after a failed query is counted as a
                    # potential eligible observation, rather than hidden.
                    if not any(v == vid for v, _ in failures):
                        if not movement_counted:
                            eligible_count += 1
                        failures.append((vid, "data_unavailable"))
            key = (tls, pid)
            self.revisions[key] += 1
            packet = LocalForecast(
                tls, pid, self.revisions[key], now, platoon.closed,
                len(platoon.member_ids), tuple(member_ids), tuple(valid_ids),
                eligible_count, min(etas) if etas else None, max(etas) if etas else None,
                "complete" if len(etas) == len(member_ids) else "partial" if etas else "unavailable",
                bool(compatible_count), tuple(failures))
            current[tls][pid] = packet
            self.seen[tls].add(pid)
            stats = self.stats[tls]
            stats["forecast_observations"] += 1
            stats["member_observations"] += len(member_ids)
            stats[f"quality_{packet.quality}"] += 1
            for _, reason in failures:
                stats[f"member_failure_{reason}"] += 1
            if packet.source_closed and packet.source_member_count >= 2:
                stats["eligible_member_observations"] += eligible_count
                stats["valid_member_observations"] += len(valid_ids)
                stats["remanent_observations"] += int(len(valid_ids) == 1)
            self.emit("local_forecast_updated", now, **asdict(packet))
        for tls in TLS_ORDER:
            for pid in sorted(self.latest[tls].keys() - current[tls].keys()):
                self.stats[tls]["retired_forecasts"] += 1
                self.emit("local_forecast_retired", now, tls_id=tls, platoon_id=pid,
                          revision=self.latest[tls][pid].revision, reason="no_local_members")
        self.latest = current
        return current

    def summary(self):
        result = {}
        for tls in TLS_ORDER:
            stats = dict(sorted(self.stats[tls].items()))
            denominator = stats.get("eligible_member_observations", 0)
            stats["valid_member_coverage"] = (stats.get("valid_member_observations", 0) / denominator
                                               if denominator else None)
            stats["unique_source_platoons_observed"] = len(self.seen[tls])
            result[tls] = stats
        return result


@dataclass(frozen=True)
class SignalSnapshot:
    program_id: str
    phase: int
    state: str
    next_switch: float
    phase_started: float


class ShadowAdvanceEvaluator:
    """Pure evaluator. Its interface cannot access TraCI or arrival outcomes."""

    def __init__(self, programs, step_seconds, horizon, emit):
        self.programs, self.step_seconds, self.horizon, self.emit = programs, step_seconds, horizon, emit
        self.previous_phase = {}
        self.entries = Counter()
        self.reserved = {}
        self.geometric = {tls: set() for tls in TLS_ORDER}
        self.before_budget = {tls: set() for tls in TLS_ORDER}
        self.reservations = {tls: [] for tls in TLS_ORDER}
        self.reasons = {tls: Counter() for tls in TLS_ORDER}
        self.all_reasons = {tls: Counter() for tls in TLS_ORDER}
        self.age_stats = {tls: Counter() for tls in TLS_ORDER}

    def seed_phases(self, signals):
        for tls, signal in signals.items():
            self.previous_phase[tls] = signal.phase
            self.entries[tls] = int(signal.phase == self.programs[tls].receptor)

    def evaluate(self, now, signals, forecasts):
        for tls in TLS_ORDER:
            signal, program = signals[tls], self.programs[tls]
            previous = self.previous_phase.get(tls, signal.phase)
            if signal.phase != previous:
                if signal.phase != (previous + 1) % len(program.states):
                    raise ValueError(f"Noncyclic observed transition for {tls}")
                if signal.phase == program.receptor:
                    self.entries[tls] += 1
            self.previous_phase[tls] = signal.phase
            if signal.program_id != program.program_id or signal.state != program.states[signal.phase]:
                raise ValueError(f"Slave signal program changed during C0: {tls}")
            occurrence = self.entries[tls] + 1
            opening = signal.next_switch
            index = (signal.phase + 1) % len(program.states)
            while index != program.receptor:
                opening += program.durations[index]
                index = (index + 1) % len(program.states)
            minimum = 36.0 if tls == "J10" else 10.0
            hypothetical_close = max(now + self.step_seconds, signal.phase_started + minimum,
                                     signal.next_switch - 5.0)
            reduction = max(0.0, signal.next_switch - hypothetical_close)
            last_reserved = self.reserved.get(tls)
            budget_used = (last_reserved[1] if last_reserved and last_reserved[0] == occurrence else 0.0)
            cooldown = bool(last_reserved and occurrence == last_reserved[0] + 1)
            common = {
                "tls_id": tls, "program": signal.program_id, "phase": signal.phase,
                "state": signal.state, "phase_started": signal.phase_started,
                "receptor_occurrence": occurrence, "nominal_opening": opening,
                "nominal_close": signal.next_switch, "hypothetical_close": hypothetical_close,
                "hypothetical_reduction": reduction, "minimum_green": minimum,
                "budget_used_before": budget_used, "budget_remaining_before": 5.0 - budget_used,
                "cooldown": cooldown, "horizon": self.horizon,
            }
            packets = sorted(forecasts[tls].values(), key=lambda f: (
                f.ready_earliest if f.ready_earliest is not None else math.inf, f.platoon_id))
            if not packets:
                self._record(now, {**common, "platoon_id": None, "revision": None,
                                    "member_ids": (), "valid_member_ids": (), "age": None,
                                    "ready_earliest": None, "ready_latest": None,
                                    "quality": "unavailable", "geometric_candidate": False,
                                    "budget_used_after": budget_used}, ["no_forecast"])
                continue
            evaluated = []
            for packet in packets:
                failures = []
                if not packet.source_closed:
                    failures.append("platoon_open")
                if packet.source_closed and packet.source_member_count < 2:
                    failures.append("singleton_original")
                age = now - packet.observed_at
                self.age_stats[tls]["observations"] += 1
                self.age_stats[tls]["age_sum"] += age
                self.age_stats[tls]["age_max"] = max(self.age_stats[tls]["age_max"], age)
                fresh = 0 <= age <= self.step_seconds
                self.age_stats[tls]["fresh"] += int(fresh)
                if not fresh:
                    failures.append("stale" if age >= 0 else "future_observation")
                if not packet.member_ids:
                    failures.append("no_local_members")
                if packet.ready_earliest is None or not packet.valid_member_ids:
                    failures.append("data_unavailable")
                if not packet.compatible:
                    failures.append("incompatible")
                if "y" in signal.state.lower():
                    failures.append("yellow")
                elif signal.phase == program.receptor:
                    failures.append("receptor_green")
                elif not any(c in signal.state for c in "Gg"):
                    failures.append("phase_not_adjustable")
                if packet.ready_earliest is not None:
                    if packet.ready_earliest > now + 10.0:
                        failures.append("availability_outside_lookahead")
                    if opening - packet.ready_earliest < 3.0:
                        failures.append("insufficient_advance")
                if reduction < self.step_seconds:
                    failures.append("minimum_green_limit" if signal.phase_started + minimum >= signal.next_switch
                                    else "insufficient_remaining_green")
                geometric = not failures
                key = (packet.platoon_id, occurrence)
                if geometric:
                    self.geometric[tls].add(key)
                if opening - reduction >= self.horizon:
                    failures.append("opening_outside_horizon")
                if not failures:
                    self.before_budget[tls].add(key)
                if last_reserved and last_reserved[0] == occurrence:
                    failures.append("budget_reserved")
                if cooldown:
                    failures.append("cooldown")
                evaluated.append([packet, age, geometric, failures])
            selected = next((packet.platoon_id for packet, _, _, failures in evaluated if not failures), None)
            for packet, age, geometric, failures in evaluated:
                if not failures and packet.platoon_id != selected:
                    failures.append("conflict_selected_other")
                record = {**common, "platoon_id": packet.platoon_id, "revision": packet.revision,
                          "observed_at": packet.observed_at, "age": age,
                          "member_ids": packet.member_ids, "valid_member_ids": packet.valid_member_ids,
                          "member_failures": packet.member_failures, "quality": packet.quality,
                          "source_member_count": packet.source_member_count,
                          "source_closed": packet.source_closed,
                          "estimate_kind": packet.estimate_kind,
                          "ready_earliest": packet.ready_earliest, "ready_latest": packet.ready_latest,
                          "geometric_candidate": geometric,
                          "budget_used_after": reduction if selected is not None else budget_used}
                if not failures:
                    self.reserved[tls] = (occurrence, reduction)
                    self.reservations[tls].append({"time": now, "platoon_id": packet.platoon_id,
                                                   "occurrence": occurrence, "seconds": reduction})
                self._record(now, record, failures)

    def _record(self, now, record, failures):
        reason = failures[0] if failures else "shadow_reserved"
        self.reasons[record["tls_id"]][reason] += 1
        self.all_reasons[record["tls_id"]].update(failures)
        self.emit("shadow_decision", now, **record,
                  result="abstain" if failures else "shadow_reserved",
                  reason=reason, failed_conditions=failures)

    def summary(self):
        result = {}
        for tls in TLS_ORDER:
            age = self.age_stats[tls]
            observations = age["observations"]
            result[tls] = {
                "reference_geometric_pairs": REFERENCE_PAIRS[tls],
                "geometric_pairs": len(self.geometric[tls]),
                "geometric_receptor_occurrences": len({occ for _, occ in self.geometric[tls]}),
                "pairs_within_horizon_before_budget": len(self.before_budget[tls]),
                "shadow_reservations": len(self.reservations[tls]),
                "hypothetical_seconds_reserved": sum(r["seconds"] for r in self.reservations[tls]),
                "reservations": self.reservations[tls],
                "mean_forecast_age": age["age_sum"] / observations if observations else None,
                "max_forecast_age": age["age_max"] if observations else None,
                "fresh_forecast_fraction": age["fresh"] / observations if observations else None,
                "primary_reasons": dict(sorted(self.reasons[tls].items())),
                "failed_conditions": dict(sorted(self.all_reasons[tls].items())),
            }
        return result


class C0Diagnostics:
    """Read-only integration; source observations precede every C0 step."""

    def __init__(self, traci, coordination_mode, horizon):
        self.mode = coordination_mode
        self.programs = read_programs(traci)
        self.step_seconds = float(traci.simulation.getDeltaT())
        self.events = []
        self.local = LocalForecastObserver(self.programs, self.emit)
        self.evaluator = (ShadowAdvanceEvaluator(self.programs, self.step_seconds, horizon, self.emit)
                          if coordination_mode == "shadow" else None)
        self.finalized = False
        if self.evaluator is not None:
            self.evaluator.seed_phases(self.read_signals(traci))
        self.emit("c0_configuration", float(traci.simulation.getTime()), mode=self.mode,
                  estimate_kind="optimistic_local_free_flow_readiness", step_seconds=self.step_seconds,
                  horizon=horizon, max_hypothetical_reduction=5,
                  minimum_green={"J2": 10, "J10": 36, "J16": 10},
                  lookahead=10, minimum_advance=3, cooldown_receptor_occurrences=1,
                  programs={tls: {"id": p.program_id, "durations": p.durations,
                                   "states": p.states, "movements": sorted(p.movements)}
                            for tls, p in self.programs.items()})

    def emit(self, event, time, **data):
        self.events.append({"event": event, "time": time, **data})

    def read_signals(self, traci):
        now = float(traci.simulation.getTime())
        return {tls: SignalSnapshot(traci.trafficlight.getProgram(tls),
                                   int(traci.trafficlight.getPhase(tls)),
                                   traci.trafficlight.getRedYellowGreenState(tls),
                                   float(traci.trafficlight.getNextSwitch(tls)),
                                   now - float(traci.trafficlight.getSpentDuration(tls)))
                for tls in TLS_ORDER}

    def observe_step(self, traci, source):
        now = float(traci.simulation.getTime())
        forecasts = self.local.observe(traci, source, now)
        if self.evaluator is not None:
            self.evaluator.evaluate(now, self.read_signals(traci), forecasts)

    def summary(self):
        return {"mode": self.mode, "semantics": "static_baseline_opportunities_only; no_slave_writes",
                "local": self.local.summary(),
                "shadow": self.evaluator.summary() if self.evaluator is not None else None}

    def finalize(self, now):
        if not self.finalized:
            self.emit("c0_summary", now, **self.summary())
            self.finalized = True
        return self.summary()

    def write_jsonl(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for event in self.events:
                stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
