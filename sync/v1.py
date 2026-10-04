"""V1: advance pending noncorridor greens from J0 green onset."""
from .common import traci, CORRIDOR_PHASE, GREEN_MIN, YELLOW

class J2Advance:
    """Shorten only pending noncorridor greens; leave SUMO's cyclic order intact."""

    def __init__(self, travel_time, tls="J2", corridor_phase=CORRIDOR_PHASE):
        self.tls = tls
        self.corridor_phase = corridor_phase
        self.noncorridor_greens = {0, 2, 4, 6} - {corridor_phase}
        self.travel_time = int(travel_time)
        self.events = []
        self.active = None
        self.queue = []
        self.future_durations = {}
        self.last_phase = traci.trafficlight.getPhase(self.tls)
        self.phase_ticks = 0
        self.phase_runs = []

    def _remaining_path(self):
        now = traci.simulation.getTime()
        current = traci.trafficlight.getPhase(self.tls)
        spent = traci.trafficlight.getSpentDuration(self.tls)
        remaining = int(round(traci.trafficlight.getNextSwitch(self.tls) - now))
        path = []
        phase = current
        while True:
            if phase == self.corridor_phase and path:
                break
            if phase == current:
                duration = remaining
                reducible = (max(0, remaining - max(1, GREEN_MIN - int(spent)))
                             if phase in self.noncorridor_greens else 0)
            else:
                duration = YELLOW if phase % 2 else 15
                reducible = duration - GREEN_MIN if phase in self.noncorridor_greens else 0
            path.append((phase, duration, reducible, phase == current))
            phase = (phase + 1) % 8
        # SUMO exposes the next phase one sampled second after nextSwitch.
        baseline_start = int(now + 1 + sum(item[1] for item in path))
        return path, baseline_start

    def _plan(self, event):
        now = traci.simulation.getTime()
        phase = traci.trafficlight.getPhase(self.tls)
        target = event["target"]
        if phase == self.corridor_phase and target <= traci.trafficlight.getNextSwitch(self.tls):
            event["status"] = "already_green_at_target"
            event["actual_start"] = int(now - traci.trafficlight.getSpentDuration(self.tls) + 1)
            event["target_error"] = 0
            event["target_window_covered"] = True
            return False
        path, baseline = self._remaining_path()
        maximum = sum(item[2] for item in path)
        requested = max(0, baseline - target)
        planned = min(requested, maximum)
        event.update({"status": "planned", "phase_at_request": phase,
                      "baseline_next_start": baseline, "requested_advance": requested,
                      "max_legal_advance": maximum, "planned_advance": planned,
                      "earliest_legal_start": baseline - maximum,
                      "legally_reachable": bool(baseline - maximum <= target <= baseline),
                      "planned_next_start": baseline - planned,
                      "green_cuts": []})
        self.future_durations = {}
        left = planned
        for idx, duration, reducible, is_current in path:
            if idx not in self.noncorridor_greens or left <= 0:
                continue
            cut = min(left, reducible)
            left -= cut
            if is_current:
                new_remaining = duration - cut
                if cut:
                    traci.trafficlight.setPhaseDuration(self.tls, new_remaining)
                total = int(round(traci.trafficlight.getSpentDuration(self.tls))) + new_remaining
            else:
                total = duration - cut
                self.future_durations[idx] = total
            event["green_cuts"].append({"phase": idx, "seconds_cut": cut,
                                        "planned_total_green": total,
                                        "applied_at": now if is_current else None})
        self.active = event
        return True

    def release(self, start):
        event = {"id": len(self.events), "J0_start": int(start),
                 "target": int(start) + self.travel_time,
                 "status": "new", "actual_start": None, "target_error": None}
        self.events.append(event)
        if self.active is not None:
            event["status"] = "queued"
            self.queue.append(event)
        else:
            self._plan(event)

    def tick(self):
        now = traci.simulation.getTime()
        phase = traci.trafficlight.getPhase(self.tls)
        if phase == self.last_phase:
            self.phase_ticks += 1
            return
        self.phase_runs.append({"phase": self.last_phase, "duration": self.phase_ticks,
                                "ended_at": now})
        self.last_phase = phase
        self.phase_ticks = 1
        if phase in self.future_durations:
            total = self.future_durations.pop(phase)
            spent = int(round(traci.trafficlight.getSpentDuration(self.tls)))
            traci.trafficlight.setPhaseDuration(self.tls, max(1, total - spent))
            if self.active:
                for cut in self.active.get("green_cuts", []):
                    if cut["phase"] == phase and cut["applied_at"] is None:
                        cut["applied_at"] = now
        if phase == self.corridor_phase:
            if self.active is not None:
                self.active["actual_start"] = int(now)
                self.active["target_error"] = int(now - self.active["target"])
                self.active["realized_advance"] = int(
                    self.active["baseline_next_start"] - now)
                self.active["target_window_covered"] = bool(
                    now <= self.active["target"] < now + 15)
                self.active["status"] = "resolved"
                self.active = None
                self.future_durations = {}
            if self.queue:
                self._plan(self.queue.pop(0))

    def finish(self):
        for event in self.events:
            if event["actual_start"] is None:
                event["status"] = "censored_at_horizon"
