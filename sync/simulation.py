"""One deterministic PPO loop for the three full-corridor controllers."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from .common import (traci, TLS, RECEIVERS, END, DEMAND, CHECKPOINT, GREEN_MIN, GREEN_MAX,
                     static_net, sumo_binary)
from .v1 import J2Advance
from .v2 import CopyGreen
from .v3 import PreannouncedWindow
from .metrics import FullObserver, validate_phases, summarize_copy, summarize_window
from policy.agent import PPOAgent
from policy.train import (_build_phase_lane_indices, _normalized_action_to_duration,
                          _structured_snapshot, _yellow_state_from_green_state)
from sumo_utils import get_green_phases

def build_agent():
    phases = get_green_phases("J0")
    if [p["index"] for p in phases] != [0, 2, 4, 6]:
        raise RuntimeError("J0 phase order changed")
    lanes = sorted({lane for p in phases for lane in p["lanes"]})
    indices = _build_phase_lane_indices(phases, lanes)
    obs_dim = len(_structured_snapshot(lanes, indices, phase_idx=0)["state"])
    agent = PPOAgent(
        obs_dim, action_dim=1, hidden_dim=256, lr=2e-4,
        gamma=0.99, gae_lambda=0.95, clip_eps=0.15, entropy_coef=0.07,
        value_coef=0.5, max_grad_norm=0.35, ppo_epochs=4,
        minibatch_size=64, device=torch.device("cpu"), normalize_obs=True,
        concentration_floor=0.2, init_action_mean=0.095,
        init_total_concentration=3.0,
    )
    metadata = agent.load(str(CHECKPOINT), map_location=torch.device("cpu"))
    if (metadata.get("junction") != "J0" or metadata.get("phase_indices") != [0, 2, 4, 6]
            or metadata.get("lanes") != lanes or metadata.get("min_green") != GREEN_MIN
            or metadata.get("max_green") != GREEN_MAX or metadata.get("obs_dim") != obs_dim
            or "yellow_sep" not in str(metadata.get("state_version", ""))):
        raise RuntimeError(f"Wrong checkpoint for J0 corrected-yellow PPO: {metadata}")
    return agent, phases, lanes, indices, metadata


def run(seed, mode, travel_times=(11, 10, 10), gui=False, delay=100):
    if seed not in range(42, 47) or mode not in ("v1", "v2", "v3"):
        raise ValueError("Expected demand 42-46 and mode v1/v2/v3")
    if len(travel_times) != 3 or any(t <= 0 for t in travel_times):
        raise ValueError("Travel times must be three positive seconds")
    if mode != "v2" and tuple(travel_times) != (11, 10, 10):
        raise ValueError("Custom travel times are supported only by V2")
    route = DEMAND / f"seed_{seed}.rou.xml"
    cumulative = (travel_times[0], sum(travel_times[:2]), sum(travel_times))
    delays = dict(zip(RECEIVERS, (12, 22, 32) if mode == "v3" else cumulative))
    with tempfile.TemporaryDirectory(prefix=f"sumo_sync_{mode}_") as tmp:
        net = Path(tmp) / "wave.net.xml"
        static_net(net)
        command = [sumo_binary(gui), "--net-file", str(net), "--route-files", str(route),
                   "--begin", "0", "--end", str(END), "--step-length", "1", "--seed", "42",
                   "--no-step-log", "true", "--no-warnings", "true"]
        if gui:
            command += ["--start", "--delay", str(delay), "--quit-on-end"]
        traci.start(command)
        try:
            programs = {t: traci.trafficlight.getProgram(t) for t in TLS}
            signatures = {t: [[(p.duration, p.state) for p in l.phases]
                              for l in traci.trafficlight.getAllProgramLogics(t)] for t in TLS}
            if set(programs.values()) != {"0"} or any(s != signatures["J0"] or len(s) != 1
                for s in signatures.values()) or sum(d for d, _ in signatures["J0"][0]) != 76:
                raise RuntimeError("Incorrect base network/programs")
            agent, phases, lanes, indices, metadata = build_agent()
            obs = FullObserver(route)
            controllers = {t: (CopyGreen(t, delays[t]) if mode == "v2" else J2Advance(delays[t], t, phase) if mode == "v1" else
                PreannouncedWindow(obs.audit, t, phase, delays[t]))
                for t, (phase, _, _) in RECEIVERS.items()}
            # Check rotated link ordering explicitly; J10 serves northbound in phase 4.
            for t, (phase, incoming, outgoing) in RECEIVERS.items():
                links = traci.trafficlight.getControlledLinks(t)
                idx = [i for i, g in enumerate(links) for link in g if link and
                       link[0].rsplit("_", 1)[0] == incoming and link[1].rsplit("_", 1)[0] == outgoing]
                if not idx or any(signatures[t][0][phase][1][i] != "G" for i in idx):
                    raise RuntimeError(f"Wrong receiver phase: {t}")
            cursor, actions, j0_runs = 0, [], []
            while traci.simulation.getTime() < END and traci.simulation.getMinExpectedNumber() > 0:
                state = _structured_snapshot(lanes, indices, phase_idx=cursor)["state"]
                action, _, _, _ = agent.act(state, deterministic=True, update_rms=False)
                decided = _normalized_action_to_duration(action, min_green=5, max_green=45)
                green = min(decided, int(END - traci.simulation.getTime()))
                start = int(traci.simulation.getTime()) + 1
                traci.trafficlight.setRedYellowGreenState("J0", phases[cursor]["state"])
                actions.append({"phase_cursor": cursor, "phase_xml": phases[cursor]["index"],
                    "decision_at": start - 1, "start": start, "green": green,
                    "decided_green": decided, "action": float(np.asarray(action).item())})
                if mode == "v3" and cursor == 0:
                    for ctrl in controllers.values():
                        ctrl.announce(start, decided)
                if cursor == 1:
                    obs.begin_release(start, green)
                    if mode == "v3":
                        for ctrl in controllers.values():
                            ctrl.prepare(start)
                for second in range(green):
                    traci.simulationStep()
                    obs.sample()
                    for ctrl in controllers.values():
                        ctrl.tick()
                    if cursor == 1 and second == 0:
                        obs.J0_releases.append(start)
                        for ctrl in controllers.values():
                            if mode == "v2":
                                ctrl.release(start, decided, start - 1)
                            else:
                                ctrl.release(start)
                j0_runs.append({"tls": "J0", "phase": phases[cursor]["index"], "start": start,
                    "duration": green, "requested": decided, "partial": green != decided})
                release_id = obs.audit.last_release_id if cursor == 1 else None
                if cursor == 1:
                    obs.end_release()
                remaining = int(END - traci.simulation.getTime())
                if remaining <= 0 or traci.simulation.getMinExpectedNumber() <= 0:
                    break
                traci.trafficlight.setRedYellowGreenState("J0",
                    _yellow_state_from_green_state(phases[cursor]["state"]))
                yellow_start = int(traci.simulation.getTime()) + 1
                yellow = min(4, remaining)
                for _ in range(yellow):
                    traci.simulationStep()
                    obs.sample()
                    for ctrl in controllers.values():
                        ctrl.tick()
                j0_runs.append({"tls": "J0", "phase": phases[cursor]["index"] + 1,
                               "start": yellow_start, "duration": yellow, "partial": yellow != 4})
                if mode == "v3" and cursor == 1 and yellow == 4:
                    for ctrl in controllers.values():
                        ctrl.release_finished(release_id)
                cursor = (cursor + 1) % 4
            obs.finish()
            for tls, ctrl in controllers.items():
                ctrl.finish()
                independent = [{k: row[k] for k in ("phase", "duration", "ended_at")}
                               for row in obs.runs[tls][:-1]]
                if independent != [{k: row[k] for k in ("phase", "duration", "ended_at")} for row in ctrl.phase_runs]:
                    raise RuntimeError(f"{tls}: controller/observer phase disagreement")
            validate_phases({"J0": j0_runs, **obs.runs})
            if mode == "v2":
                copied_runs = [{k: v for k, v in r.items() if k not in ("tls", "partial")} | {"censored": r["partial"]} for r in j0_runs]
                return summarize_copy(seed, route, obs, controllers, actions, copied_runs, programs, metadata, travel_times, delays)
            return summarize_window(seed, mode, route, obs, controllers, actions, j0_runs, programs, delays)
        finally:
            traci.close()
