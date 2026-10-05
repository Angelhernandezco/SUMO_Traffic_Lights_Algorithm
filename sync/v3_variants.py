"""Small, separately selectable V3 experiments; original V3 stays intact."""
from .common import traci, GREEN_MIN, GREEN_MAX, RECEIVERS
from .v3 import PreannouncedWindow


class QueueBalancedWindow(PreannouncedWindow):
    """V3.1: allocate the same advance to less queued pending greens in J2."""

    balance_all_receivers = False

    def __init__(self, audit, tls, corridor_phase, travel_time):
        super().__init__(audit, tls, corridor_phase, travel_time)
        links = traci.trafficlight.getControlledLinks(tls)
        phases = traci.trafficlight.getAllProgramLogics(tls)[0].phases
        self.phase_lanes = {
            phase: sorted({link[0] for i, group in enumerate(links)
                           if phases[phase].state[i] in "Gg"
                           for link in group if link})
            for phase in self.noncorridor_greens
        }

    def _remaining_path(self):
        path, baseline = super()._remaining_path()
        if self.tls != "J2" and not self.balance_all_receivers:
            return path, baseline
        # Sorting changes cut allocation only. SUMO still executes every phase
        # in its original cyclic order; the total requested advance is unchanged.
        pressure = {phase: (
            sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes),
            sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes))
            for phase, lanes in self.phase_lanes.items()}
        ranked = sorted(enumerate(path), key=lambda item: (
            pressure.get(item[1][0], (float("inf"), float("inf"))), item[0]))
        return [item for _, item in ranked], baseline


class TailClosedWindow(QueueBalancedWindow):
    """V3.2: return unused J2 green after the real release has cleared E1."""

    def tick(self):
        super().tick()
        if self.tls != "J2" or traci.trafficlight.getPhase(self.tls) != self.corridor_phase:
            return
        now = int(traci.simulation.getTime())
        spent = int(round(traci.trafficlight.getSpentDuration(self.tls)))
        remaining = int(round(traci.trafficlight.getNextSwitch(self.tls) - now))
        if spent < GREEN_MIN or remaining <= 1:
            return
        attached = [e for e in self.events if e.get("actual_start") == self.current_green_start]
        if not attached or any(not e["release_finished"] for e in attached):
            return
        if any(e["crossings"] and now < e["tail_target"] for e in attached):
            return  # Keep every sampled second of the causal target window.
        if any(not e["closed"] and e not in attached for e in self.events):
            return  # A new announcement must retain its preparation opportunity.
        incoming = RECEIVERS[self.tls][1]
        if traci.edge.getLastStepVehicleNumber(incoming):
            return  # Protect local/turning traffic too, not only measured platoons.
        traci.trafficlight.setPhaseDuration(self.tls, 1)
        for event in attached:
            event["corridor_early_close_at"] = now
            event["unused_corridor_seconds_removed"] = remaining - 1


class DeferredQueueWindow(QueueBalancedWindow):
    """V3.3: retain the notice, commit J2 cuts at the last legal opportunity."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deferred = []

    def _feasibility(self, event):
        now = int(traci.simulation.getTime())
        phase = traci.trafficlight.getPhase(self.tls)
        if phase == self.corridor_phase:
            start = self.current_green_start
            if start is None:
                start = int(now - traci.trafficlight.getSpentDuration(self.tls) + 1)
            if start <= event["front_target"] < start + GREEN_MAX:
                return start, start, True
        path, baseline = self._remaining_path()
        earliest = baseline - sum(p[2] for p in path)
        planned = max(earliest, min(baseline, event["front_target"]))
        return earliest, planned, planned <= event["front_target"] < planned + GREEN_MAX

    def _request(self, event, reason):
        if self.tls == "J2" and reason == "preannouncement" and self.active is None:
            earliest, planned, reachable = self._feasibility(event)
            event.update(status="deferred_preannouncement", earliest_legal_start=earliest,
                         planned_next_start=planned, front_legal_at_request=reachable,
                         deferred_at=int(traci.simulation.getTime()))
            self.deferred.append(event)
            return
        self.deferred = [item for item in self.deferred if item is not event]
        super()._request(event, reason)

    def tick(self):
        # Ingest crossings first: a real front can replace the provisional one
        # before a cut is committed in this sampled second.
        super().tick()
        now = int(traci.simulation.getTime())
        phase = traci.trafficlight.getPhase(self.tls)
        for event in list(self.deferred):
            if event["closed"]:
                self.deferred.remove(event)
                continue
            earliest, planned, _ = self._feasibility(event)
            if phase == self.corridor_phase and planned == self.current_green_start:
                commit = (now >= event["front_target"] or
                          now + 1 >= traci.trafficlight.getNextSwitch(self.tls))
            else:
                commit = earliest >= event["front_target"] - 1
            if commit:
                self.deferred.remove(event)
                event["cuts_committed_at"] = now
                event["notice_to_commit_s"] = now - event["announced_at"]
                super()._request(event, "latest_legal_preparation")

    def _close(self, event, status):
        self.deferred = [item for item in self.deferred if item is not event]
        super()._close(event, status)


class FullQueueBalancedWindow(QueueBalancedWindow):
    """V3.4: apply V3.1's queue-aware cut allocation to all three receivers."""

    balance_all_receivers = True


class AlignedQueueWindow(FullQueueBalancedWindow):
    """V3.5: avoid spending a corridor green well before the expected front."""

    def _plan(self, event):
        now = int(traci.simulation.getTime())
        phase = traci.trafficlight.getPhase(self.tls)
        if phase == self.corridor_phase:
            start = self.current_green_start
            if start is None:
                start = int(now - traci.trafficlight.getSpentDuration(self.tls) + 1)
            if event["front_target"] < start + GREEN_MAX and event["tail_target"] >= start:
                return super()._plan(event)
        path, baseline = self._remaining_path()
        if event["front_target"] < baseline + 15:
            return super()._plan(event)
        spent = int(round(traci.trafficlight.getSpentDuration(self.tls)))
        capacity = {idx: max(0, GREEN_MAX - duration - (spent if current else 0))
                    for idx, duration, _, current in path if idx in self.noncorridor_greens}
        requested = max(0, event["front_target"] - 1 - baseline)
        delay = min(requested, sum(capacity.values()))
        if not delay:
            return super()._plan(event)
        old_cuts = event.get("green_cuts", [])
        self._cancel_pending_cuts(event)
        self.future_durations = {}
        new_cuts, left = [], delay
        # Reverse the queue ranking: give extra service to the busiest pending
        # greens. This delays an early opening, never changes phase order.
        for idx, duration, _, current in reversed(path):
            amount = min(left, capacity.get(idx, 0))
            if not amount:
                continue
            left -= amount
            if current:
                traci.trafficlight.setPhaseDuration(self.tls, duration + amount)
                total = spent + duration + amount
            else:
                total = duration + amount
                self.future_durations[idx] = total
            new_cuts.append({"phase": idx, "seconds_cut": 0, "seconds_added": amount,
                             "planned_total_green": total,
                             "applied_at": now if current else None,
                             "generation": len(event["plans"]), "cancelled": False})
        opening = baseline + delay
        event.update(status="planned", phase_at_request=phase,
                     baseline_next_start=baseline, requested_advance=0,
                     max_legal_advance=sum(p[2] for p in path), planned_advance=-delay,
                     earliest_legal_start=baseline-sum(p[2] for p in path),
                     latest_legal_start=baseline+sum(capacity.values()),
                     legally_reachable=opening <= event["front_target"] < opening+GREEN_MAX,
                     front_legal_at_request=opening <= event["front_target"] < opening+GREEN_MAX,
                     planned_next_start=opening, green_cuts=old_cuts+new_cuts,
                     requested_delay=requested, planned_delay=delay)
        event["plans"].append({"at": now, "reason": "avoid_early_opening",
                              "front": event["front_target"], "tail": event["tail_target"],
                              "earliest": event["earliest_legal_start"], "opening": opening,
                              "reachable": event["front_legal_at_request"],
                              "cuts": [dict(c) for c in new_cuts]})
        self.active = event
        return True


class ArrivalGuardWindow(AlignedQueueWindow):
    """V3.6: prepare one sampled second before the unchanged causal front."""

    opening_guard_s = 1

    def _plan(self, event):
        original_target = event["target"]
        event["target"] = original_target - self.opening_guard_s
        event["opening_guard_s"] = self.opening_guard_s
        try:
            return super()._plan(event)
        finally:
            # Observations, front/tail validation and future updates keep +12.
            event["target"] = original_target


class DoubleSampleGuardWindow(ArrivalGuardWindow):
    """V3.7: account for sampled crossing/arrival uncertainty at both ends."""

    opening_guard_s = 2


WINDOW_CONTROLLERS = {"v3": PreannouncedWindow, "v3.1": QueueBalancedWindow,
                      "v3.2": TailClosedWindow, "v3.3": DeferredQueueWindow,
                      "v3.4": FullQueueBalancedWindow, "v3.5": AlignedQueueWindow,
                      "v3.6": ArrivalGuardWindow, "v3.7": DoubleSampleGuardWindow}
