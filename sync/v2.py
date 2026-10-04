"""V2: copy the decided PPO green duration to every receiver."""
from .common import traci, RECEIVERS, GREEN_MIN, GREEN_MAX

class CopyGreen:
    """Adjust pending non-corridor greens; execute one complete copied green."""

    def __init__(self, tls, travel_time):
        self.tls = tls
        self.travel_time = int(travel_time)
        self.corridor_phase, incoming, outgoing = RECEIVERS[tls]
        self.noncorridor = {0, 2, 4, 6} - {self.corridor_phase}
        logic = traci.trafficlight.getAllProgramLogics(tls)[0]
        self.durations = [int(p.duration) for p in logic.phases]
        links = traci.trafficlight.getControlledLinks(tls)
        indices = [i for i, group in enumerate(links) for link in group if link
                   and link[0].rsplit("_", 1)[0] == incoming
                   and link[1].rsplit("_", 1)[0] == outgoing]
        if not indices or any(logic.phases[self.corridor_phase].state[i] != "G" for i in indices):
            raise RuntimeError(f"Wrong corridor movement/phase for {tls}")
        self.events = []
        self.active = None
        self.serving = None
        self.queue = []
        self.future_durations = {}
        self.last_phase = traci.trafficlight.getPhase(tls)
        self.initial_phase_partial = (
            traci.trafficlight.getSpentDuration(tls) > 0 or
            round(traci.trafficlight.getNextSwitch(tls) - traci.simulation.getTime())
            != self.durations[self.last_phase])
        self.phase_ticks = 0
        self.phase_runs = []

    def _remaining_path(self):
        now = int(traci.simulation.getTime())
        current = traci.trafficlight.getPhase(self.tls)
        spent = int(round(traci.trafficlight.getSpentDuration(self.tls)))
        remaining = int(round(traci.trafficlight.getNextSwitch(self.tls) - now))
        path = []
        phase = current
        while True:
            if phase == self.corridor_phase and path:
                break
            duration = remaining if phase == current else self.durations[phase]
            reducible = expandable = 0
            if phase in self.noncorridor:
                elapsed = spent if phase == current else 0
                minimum_remaining = max(1, GREEN_MIN - elapsed) if phase == current else GREEN_MIN
                reducible = max(0, duration - minimum_remaining)
                expandable = max(0, GREEN_MAX - elapsed - duration)
            path.append((phase, duration, reducible, expandable, phase == current))
            phase = (phase + 1) % 8
        # Same 1 s sampling convention as V1: next phase appears after nextSwitch.
        return path, now + 1 + sum(item[1] for item in path)

    def _copy_into_current_green(self, event, start):
        spent = int(round(traci.trafficlight.getSpentDuration(self.tls)))
        green = event["J0_green"]
        if spent >= green:
            raise RuntimeError("Cannot retroactively copy an elapsed green")
        traci.trafficlight.setPhaseDuration(self.tls, green - spent)
        event.update(actual_start=start, target_error=start - event["target"],
                     receiver_green_planned=green, status="serving_copy")
        self.serving = event

    def _plan(self, event):
        now = int(traci.simulation.getTime())
        phase = traci.trafficlight.getPhase(self.tls)
        spent = int(round(traci.trafficlight.getSpentDuration(self.tls)))
        start = now - spent + 1
        # An unassigned green can serve the front only if its complete duration
        # can still be made exactly equal to this PPO action. Never merge copies.
        if (phase == self.corridor_phase and self.serving is None
                and spent < event["J0_green"]
                and start <= event["target"] < start + event["J0_green"]):
            event.update(start_target_reachable=start == event["target"],
                         front_green_feasible=True, using_current_green=True,
                         earliest_legal_start=start, adjustments=[])
            self._copy_into_current_green(event, start)
            return
        path, baseline = self._remaining_path()
        cut_capacity = sum(item[2] for item in path)
        extension_capacity = sum(item[3] for item in path)
        requested_shift = event["target"] - baseline
        shift = max(-cut_capacity, min(extension_capacity, requested_shift))
        opening = baseline + shift
        event.update(status="planned", phase_at_request=phase,
                     baseline_next_start=baseline, requested_shift=requested_shift,
                     planned_shift=shift, earliest_legal_start=baseline - cut_capacity,
                     latest_legal_start=baseline + extension_capacity,
                     start_target_reachable=opening == event["target"],
                     front_green_feasible=opening <= event["target"] < opening + event["J0_green"],
                     planned_next_start=opening, adjustments=[])
        self.future_durations = {}
        left = abs(shift)
        for idx, duration, reducible, expandable, is_current in path:
            capacity = reducible if shift < 0 else expandable
            if idx not in self.noncorridor or left <= 0:
                continue
            amount = min(left, capacity)
            if not amount:
                continue
            left -= amount
            change = -amount if shift < 0 else amount
            if is_current:
                remaining = duration + change
                traci.trafficlight.setPhaseDuration(self.tls, remaining)
                total = spent + remaining
            else:
                total = duration + change
                self.future_durations[idx] = total
            event["adjustments"].append({"phase": idx, "seconds_change": change,
                                         "planned_total_green": total,
                                         "applied_at": now if is_current else None})
        self.active = event

    def release(self, start, green, decision_at):
        if not GREEN_MIN <= green <= GREEN_MAX:
            raise ValueError("The copied PPO green must be between 5 and 45 s")
        event = {"id": len(self.events), "tls": self.tls, "J0_start": int(start),
                 "J0_decision_at": int(decision_at), "J0_green": int(green),
                 "requested_at": int(traci.simulation.getTime()),
                 "target": int(start) + self.travel_time,
                 "target_end": int(start) + self.travel_time + int(green),
                 "actual_start": None, "actual_end": None, "target_error": None,
                 "status": "new", "using_current_green": False, "adjustments": []}
        self.events.append(event)
        if self.active is not None:
            event["status"] = "queued"
            self.queue.append(event)
        else:
            self._plan(event)

    def tick(self):
        now = int(traci.simulation.getTime())
        phase = traci.trafficlight.getPhase(self.tls)
        if phase == self.last_phase:
            self.phase_ticks += 1
            return
        self.phase_runs.append({"phase": self.last_phase, "duration": self.phase_ticks,
                                "ended_at": now,
                                "partial": not self.phase_runs and self.initial_phase_partial})
        leaving = self.last_phase == self.corridor_phase
        self.last_phase = phase
        self.phase_ticks = 1
        if leaving and self.serving is not None:
            event = self.serving
            event["actual_end"] = now
            event["receiver_green_executed"] = now - event["actual_start"]
            if event["receiver_green_executed"] != event["J0_green"]:
                raise RuntimeError(f"{self.tls}: completed green differs from copied PPO duration")
            event["status"] = "copied"
            self.serving = None
        if phase in self.future_durations:
            total = self.future_durations.pop(phase)
            spent = int(round(traci.trafficlight.getSpentDuration(self.tls)))
            traci.trafficlight.setPhaseDuration(self.tls, total - spent)
            if self.active:
                for adjustment in self.active["adjustments"]:
                    if adjustment["phase"] == phase and adjustment["applied_at"] is None:
                        adjustment["applied_at"] = now
        if phase == self.corridor_phase:
            if self.active is not None:
                event = self.active
                self._copy_into_current_green(event, now)
                self.active = None
                self.future_durations = {}
            if self.queue:
                self._plan(self.queue.pop(0))

    def finish(self):
        now = int(traci.simulation.getTime())
        if self.serving is not None:
            self.serving.update(actual_end=now + 1, copy_censored=True,
                                status="copy_censored_at_horizon")
        for event in self.events:
            if event["actual_start"] is None:
                event["status"] = "unserved_at_horizon"
            start, stop = event["actual_start"], event["actual_end"]
            event["window_coverage_pct"] = (100 * max(0, min(stop, event["target_end"]) -
                max(start, event["target"])) / event["J0_green"] if start is not None else 0.0)
            event["start_delay_s"] = max(0, start - event["target"]) if start is not None else None
        self.partial_run = {"phase": self.last_phase, "duration": self.phase_ticks,
                            "ended_at": now + 1, "partial": True}
