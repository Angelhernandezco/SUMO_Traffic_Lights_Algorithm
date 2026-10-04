"""V3: preannounce, correct the front and extend the causal tail."""
from .common import traci, CORRIDOR_PHASE, FLIGHT_TIME, GREEN_MAX, YELLOW, END
from .v1 import J2Advance

class _CausalWindow(J2Advance):
    """Shared causal window operations used by the preannouncement controller."""

    def __init__(self, audit, tls="J2", corridor_phase=CORRIDOR_PHASE,
                 travel_time=FLIGHT_TIME):
        super().__init__(travel_time, tls, corridor_phase)
        self.audit = audit
        self.next_crossing = 0
        self.current_green_start = None
        self.current_green_events = []

    def prepare(self, start):
        path, baseline_start = self._remaining_path()
        event = {"id": len(self.events), "J0_start": int(start),
                 "prepared_at": int(traci.simulation.getTime()),
                 "J2_phase_at_prepare": traci.trafficlight.getPhase(self.tls),
                 "earliest_possible_front": int(start) + self.travel_time,
                 "baseline_next_start_at_prepare": baseline_start,
                 "pending_phases_at_prepare": [item[0] for item in path],
                 "first_cross": None, "last_cross": None, "front_target": None,
                 "tail_target": None, "target": None, "crossings": [],
                 "status": "awaiting_first_vehicle", "actual_start": None,
                 "actual_end": None, "target_error": None, "green_cuts": []}
        self.events.append(event)

    def release(self, start):
        if not self.events or self.events[-1]["J0_start"] != int(start):
            # Standalone use; the experiment normally prepares at green onset.
            self.prepare(start)
        if self.events[-1]["J0_start"] != int(start):
            raise RuntimeError("Prepared J0 release does not match PPO green")
        self._ingest_crossings()

    def _attach_current_green(self, event, start):
        event["actual_start"] = int(start)
        event["target_error"] = int(start - event["front_target"])
        event["front_legal_at_request"] = bool(
            start <= event["front_target"] < start + GREEN_MAX)
        event["status"] = "using_current_green"
        if event not in self.current_green_events:
            self.current_green_events.append(event)
        self._extend_current_green()

    def _extend_current_green(self):
        if traci.trafficlight.getPhase(self.tls) != self.corridor_phase:
            return
        now = traci.simulation.getTime()
        start = self.current_green_start
        if start is None:
            start = int(now - traci.trafficlight.getSpentDuration(self.tls) + 1)
            self.current_green_start = start
        spent = int(round(traci.trafficlight.getSpentDuration(self.tls)))
        remaining = int(round(traci.trafficlight.getNextSwitch(self.tls) - now))
        current_total = spent + remaining
        required = max(event["tail_target"] - start + 1
                       for event in self.current_green_events)
        desired = min(GREEN_MAX, max(current_total, required))
        if desired > current_total:
            traci.trafficlight.setPhaseDuration(self.tls, desired - spent)
        for event in self.current_green_events:
            event["required_corridor_green"] = event["tail_target"] - start + 1
            event["J2_green_planned"] = desired
            event["tail_beyond_45"] = event["tail_target"] >= start + GREEN_MAX

    def _plan(self, event):
        now = traci.simulation.getTime()
        if traci.trafficlight.getPhase(self.tls) == self.corridor_phase:
            start = self.current_green_start
            if start is None:
                start = int(now - traci.trafficlight.getSpentDuration(self.tls) + 1)
                self.current_green_start = start
            if (event["front_target"] < start + GREEN_MAX
                    and event["tail_target"] >= start):
                self._attach_current_green(event, start)
                return False
        planned = super()._plan(event)
        if event.get("planned_next_start") is not None:
            start = event["planned_next_start"]
            event["front_legal_at_request"] = bool(
                start <= event["front_target"] < start + GREEN_MAX)
        return planned


class PreannouncedWindow(_CausalWindow):
    """One event per J0 release, prepared from the preceding decided action."""

    def announce(self, previous_start, decided_green):
        straight_start = previous_start + decided_green + YELLOW
        super().prepare(straight_start)
        event = self.events[-1]
        event.update({
            "announced_at": int(traci.simulation.getTime()),
            "previous_green_start": previous_start,
            "previous_decided_green": decided_green,
            "announced_J0_start": straight_start,
            "actual_J0_start": None, "release_finished": False,
            "provisional_front": straight_start + self.travel_time,
            "front_target": straight_start + self.travel_time,
            "tail_target": straight_start + self.travel_time,
            "target": straight_start + self.travel_time,
            "plans": [], "openings": [], "closed_at": None,
            "first_cross_updates": 0, "closed": False,
        })
        if event["target"] >= END:
            event.update(status="censored_announcement", closed=True,
                         closed_at=int(traci.simulation.getTime()))
            return
        self._request(event, "preannouncement")
        event["front_reachable_at_announcement"] = event.get("front_legal_at_request")
        event["earliest_start_at_announcement"] = event.get("earliest_legal_start")
        event["planned_start_at_announcement"] = event.get("planned_next_start",
                                                                   event.get("actual_start"))

    def prepare(self, start):
        event = self.events[self.audit.last_release_id]
        if event["announced_J0_start"] != start:
            raise RuntimeError("Preannouncement and actual J0 green start differ")
        event["actual_J0_start"] = start

    def release_finished(self, release_id):
        event = self.events[release_id]
        event["release_finished"] = True
        event["release_finished_at"] = int(traci.simulation.getTime())
        self._close_finished()

    def _cancel_pending_cuts(self, event):
        for cut in event.get("green_cuts", []):
            if cut["applied_at"] is None and not cut.get("cancelled"):
                cut["cancelled"] = True
        if self.active is event:
            self.future_durations = {}
            self.active = None

    def _request(self, event, reason):
        event["request_reason"] = reason
        if self.active is not None and self.active is not event:
            if event not in self.queue:
                self.queue.append(event)
            event["status"] = "queued"
            event["queued_at"] = int(traci.simulation.getTime())
            return
        self._plan(event)

    def _plan(self, event):
        # Recompute only unapplied cuts after the real front corrects the notice.
        # A cut already made to the current green is never restored or repeated.
        old_cuts = event.get("green_cuts", [])
        self._cancel_pending_cuts(event)
        event["green_cuts"] = []
        self.future_durations = {}
        planned = super()._plan(event)
        new_cuts = event.get("green_cuts", [])
        generation = len(event["plans"])
        for cut in new_cuts:
            cut["generation"] = generation
            cut["cancelled"] = False
        event["green_cuts"] = old_cuts + new_cuts
        event["plans"].append({
            "at": int(traci.simulation.getTime()),
            "reason": event.get("request_reason", "queued_request"),
            "front": event["front_target"], "tail": event["tail_target"],
            "earliest": event.get("earliest_legal_start"),
            "opening": (event.get("planned_next_start") if planned
                        else event.get("actual_start")),
            "reachable": event.get("front_legal_at_request"),
            "cuts": [dict(cut) for cut in new_cuts],
        })
        return planned

    def _attach_current_green(self, event, start):
        super()._attach_current_green(event, start)
        if not event["openings"] or event["openings"][-1]["start"] != start:
            event["openings"].append({"start": start, "end": None,
                                      "front_at_attachment": event["front_target"]})

    def _ingest_crossings(self):
        while self.next_crossing < len(self.audit.crossings):
            crossing = self.audit.crossings[self.next_crossing]
            if crossing["release_id"] >= len(self.events):
                raise RuntimeError("Crossing has no announced event")
            self.next_crossing += 1
            event = self.events[crossing["release_id"]]
            event["crossings"].append(crossing)
            at = crossing["time"]
            if event["first_cross"] is None:
                event["first_cross_updates"] += 1
                event["first_cross"] = at
                event["last_cross"] = at
                event["front_target"] = event["tail_target"] = at + self.travel_time
                event["target"] = event["front_target"]
                event["front_correction_s"] = event["front_target"] - event["provisional_front"]
                if event["front_target"] >= END:
                    # Observe a terminal release without reopening an expired
                    # preparation or making cuts for an unobservable arrival.
                    self._close(event, "censored_window")
                    continue
                if event["closed"]:
                    event["cross_after_close"] = True
                    event["closed"] = False
                    event["closed_at"] = None
                if (event in self.current_green_events and
                        event["front_target"] < self.current_green_start + GREEN_MAX):
                    self._extend_current_green()
                else:
                    if event in self.current_green_events:
                        self.current_green_events.remove(event)
                    self._request(event, "first_cross_correction")
                event["front_reachable_at_first_cross"] = event.get("front_legal_at_request")
            else:
                event["last_cross"] = max(event["last_cross"], at)
                event["tail_target"] = event["last_cross"] + self.travel_time
                if event in self.current_green_events:
                    self._extend_current_green()

    def _close(self, event, status):
        self._cancel_pending_cuts(event)
        self.queue = [item for item in self.queue if item is not event]
        if event in self.current_green_events:
            self.current_green_events.remove(event)
        event.update(closed=True, closed_at=int(traci.simulation.getTime()), status=status)

    def _close_finished(self):
        now = int(traci.simulation.getTime())
        for event in self.events:
            if event["closed"] or not event["release_finished"]:
                continue
            if not event["crossings"]:
                self._close(event, "empty_release")
            elif event["actual_start"] is not None and now > max(
                    event["tail_target"], event["actual_start"]):
                self._close(event, "served_late" if event["actual_start"] >
                            event["front_target"] else "served_window")

    def tick(self):
        now = int(traci.simulation.getTime())
        phase = traci.trafficlight.getPhase(self.tls)
        entering = phase == self.corridor_phase and self.last_phase != phase
        leaving = self.last_phase == self.corridor_phase and phase != self.last_phase
        prior_active = self.active if entering else None
        prior_current = list(self.current_green_events) if leaving else []
        if entering:
            self.current_green_start = now
        # Same duration application and cyclic phase tracking as V1/V2.
        # Cancelled cuts remain in the audit, but cannot be marked as applied.
        if phase != self.last_phase and phase in self.future_durations and self.active:
            for cut in self.active.get("green_cuts", []):
                if cut.get("cancelled") and cut["applied_at"] is None:
                    cut["applied_at"] = "cancelled"
        J2Advance.tick(self)
        if entering and prior_active is not None and prior_active["actual_start"] == now:
            self._attach_current_green(prior_active, now)
        if leaving:
            for event in prior_current:
                event["actual_end"] = now
            self.current_green_events = []
            self.current_green_start = None
        for event in self.events:
            if event["openings"] and event["openings"][-1]["end"] is None:
                if phase != self.corridor_phase:
                    event["openings"][-1]["end"] = now
        self._ingest_crossings()
        self._close_finished()

    def finish(self):
        now = int(traci.simulation.getTime())
        for event in self.events:
            if event["openings"] and event["openings"][-1]["end"] is None:
                event["openings"][-1].update(end=now + 1, censored=True)
            if not event["closed"]:
                self._close(event, "censored_at_horizon")
