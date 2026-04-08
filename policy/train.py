import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import traci
from sumolib import checkBinary

from sumo_utils import (
    get_green_phases,
    get_vehicle_numbers,
    get_waiting_time,
    set_phase_by_index,
)
from policy.agent import PPOAgent

EPS = 1e-6


def _get_lane_halting_numbers(lanes: List[str]) -> Dict[str, float]:
    return {lane: float(traci.lane.getLastStepHaltingNumber(lane)) for lane in lanes}


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den + EPS)


def _phase_vehicle_demands(vehicles_per_lane: Dict[str, float], phases: list) -> np.ndarray:
    return np.array(
        [sum(float(vehicles_per_lane.get(lane, 0.0)) for lane in phase["lanes"]) for phase in phases],
        dtype=np.float32,
    )


def _phase_queue_demands(halts_per_lane: Dict[str, float], phases: list) -> np.ndarray:
    return np.array(
        [sum(float(halts_per_lane.get(lane, 0.0)) for lane in phase["lanes"]) for phase in phases],
        dtype=np.float32,
    )


def _phase_pressures(phase_vehicle: np.ndarray, phase_queue: np.ndarray) -> np.ndarray:
    return phase_queue + 0.50 * phase_vehicle


def _dominance_features(phase_pressure: np.ndarray, phase_idx: int) -> np.ndarray:
    active = float(phase_pressure[phase_idx])
    total = float(phase_pressure.sum())
    others = np.delete(phase_pressure, phase_idx)
    other_total = float(others.sum()) if others.size > 0 else 0.0
    other_max = float(others.max()) if others.size > 0 else 0.0
    other_mean = float(others.mean()) if others.size > 0 else 0.0
    next_idx = (int(phase_idx) + 1) % int(len(phase_pressure))
    next_pressure = float(phase_pressure[next_idx])
    max_all = float(phase_pressure.max())
    active_share = _safe_ratio(active, total)
    active_vs_total_rest = _safe_ratio(active, other_total)
    active_vs_other_max = _safe_ratio(active, other_max)
    active_vs_other_mean = _safe_ratio(active, other_mean)
    active_vs_next = _safe_ratio(active, next_pressure)
    active_minus_other_max = active - other_max
    active_minus_other_mean = active - other_mean
    active_is_top_soft = _safe_ratio(active, max_all)
    dominance_margin_norm = _safe_ratio(active_minus_other_max, total)

    return np.array(
        [
            np.log1p(active),
            np.log1p(other_total),
            np.log1p(other_max),
            np.log1p(other_mean),
            np.log1p(total),
            active_share,
            active_vs_total_rest,
            active_vs_other_max,
            active_vs_other_mean,
            active_vs_next,
            active_minus_other_max,
            active_minus_other_mean,
            active_is_top_soft,
            dominance_margin_norm,
        ],
        dtype=np.float32,
    )


def _build_observation(
    vehicles_per_lane: Dict[str, float],
    halts_per_lane: Dict[str, float],
    lanes: List[str],
    phases: list,
    *,
    phase_idx: int,
) -> np.ndarray:
    if not (0 <= int(phase_idx) < len(phases)):
        raise ValueError(f"phase_idx out of range: {phase_idx}")

    lane_vehicle = np.array([float(vehicles_per_lane.get(lane, 0.0)) for lane in lanes], dtype=np.float32)
    lane_halts = np.array([float(halts_per_lane.get(lane, 0.0)) for lane in lanes], dtype=np.float32)

    phase_vehicle = _phase_vehicle_demands(vehicles_per_lane, phases)
    phase_queue = _phase_queue_demands(halts_per_lane, phases)
    phase_pressure = _phase_pressures(phase_vehicle, phase_queue)
    pressure_total = float(phase_pressure.sum())
    pressure_share = phase_pressure / float(pressure_total + EPS)

    phase_one_hot = np.zeros((len(phases),), dtype=np.float32)
    phase_one_hot[int(phase_idx)] = 1.0

    dominance = _dominance_features(phase_pressure, int(phase_idx))

    return np.concatenate(
        [
            np.log1p(lane_vehicle),
            np.log1p(lane_halts),
            phase_one_hot,
            np.log1p(phase_vehicle),
            np.log1p(phase_queue),
            pressure_share.astype(np.float32),
            dominance,
        ],
        axis=0,
    )


def _obs_dim(num_lanes: int, num_phases: int) -> int:
    return 2 * num_lanes + 5 * num_phases + 14


def _normalized_action_to_duration(action01: np.ndarray, *, min_green: int, max_green: int) -> int:
    a = np.asarray(action01, dtype=np.float32)

    if a.ndim == 0:
        a01 = float(a)
    elif a.ndim == 1 and a.shape[0] == 1:
        a01 = float(a[0])
    else:
        raise ValueError(f"action01 must be a scalar or shape (1,), got shape={a.shape}")

    if int(max_green) < int(min_green):
        raise ValueError(f"max_green={max_green} must be >= min_green={min_green}")

    a01 = float(np.clip(a01, 0.0, 1.0))
    span = int(max_green) - int(min_green)

    # Mapeo más suave que 2.5: todavía favorece el mínimo en acciones medias,
    # pero sin castigar tanto como para dejar casi todo pegado abajo.
    shaped = a01 ** 1.5

    duration = int(min_green) + int(np.rint(shaped * span))
    return max(int(min_green), int(duration))


def _select_junction_phases_and_lanes(max_phases: int = 4) -> Tuple[str, list, list]:
    junctions = traci.trafficlight.getIDList()
    if not junctions:
        raise RuntimeError("No traffic lights found in the SUMO network.")

    junction = junctions[0]
    green_phases = get_green_phases(junction)
    if not green_phases:
        raise RuntimeError(f"No green phases detected for junction {junction}.")

    if len(green_phases) > max_phases:
        green_phases = green_phases[:max_phases]

    lanes = sorted({lane for phase in green_phases for lane in phase["lanes"]})
    if not lanes:
        raise RuntimeError("No controlled lanes extracted from green phases.")

    return junction, green_phases, lanes


class SumoTrafficEnv:
    def __init__(
        self,
        lanes: List[str],
        phases: list,
        junction: str,
        *,
        min_green: int,
        max_green: int,
    ) -> None:
        self.lanes = lanes
        self.phases = phases
        self.junction = junction
        self.min_green = int(min_green)
        self.max_green = int(max_green)
        self.phase_cursor = 0
        self.observation_dim = _obs_dim(len(self.lanes), len(self.phases))

    def reset(self) -> np.ndarray:
        self.phase_cursor = 0
        return self._obs()

    def _current_measurements(self) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        vehicles_per_lane = get_vehicle_numbers(self.lanes)
        halts_per_lane = _get_lane_halting_numbers(self.lanes)
        phase_vehicle = _phase_vehicle_demands(vehicles_per_lane, self.phases)
        phase_queue = _phase_queue_demands(halts_per_lane, self.phases)
        phase_pressure = _phase_pressures(phase_vehicle, phase_queue)
        return vehicles_per_lane, halts_per_lane, phase_vehicle, phase_queue, phase_pressure

    def _obs(self) -> np.ndarray:
        vehicles_per_lane, halts_per_lane, _, _, _ = self._current_measurements()
        return _build_observation(
            vehicles_per_lane,
            halts_per_lane,
            self.lanes,
            self.phases,
            phase_idx=int(self.phase_cursor),
        )

    def step(self, action01: np.ndarray, *, max_steps: int) -> Tuple[np.ndarray, float, bool, int, float, dict]:
        if int(max_steps) <= 0:
            done = bool(traci.simulation.getMinExpectedNumber() <= 0)
            return self._obs(), 0.0, done, 0, 0.0, {}

        current_phase_idx = int(self.phase_cursor)
        current_phase = self.phases[current_phase_idx]
        active_lanes = current_phase["lanes"]

        requested_duration = _normalized_action_to_duration(
            action01,
            min_green=int(self.min_green),
            max_green=int(self.max_green),
        )
        requested_duration = min(int(requested_duration), int(max_steps))

        (
            vehicles_before,
            halts_before,
            phase_vehicle_before,
            phase_queue_before,
            phase_pressure_before,
        ) = self._current_measurements()

        active_queue_before = float(phase_queue_before[current_phase_idx])
        active_pressure_before = float(phase_pressure_before[current_phase_idx])
        other_pressure_before = np.delete(phase_pressure_before, current_phase_idx)
        other_max_before = float(other_pressure_before.max()) if other_pressure_before.size > 0 else 0.0
        total_queue_before = float(phase_queue_before.sum())
        active_wait_before = float(get_waiting_time(active_lanes))

        waiting_sum = 0.0
        elapsed = 0
        idle_green_seconds = 0

        if requested_duration > 0:
            set_phase_by_index(self.junction, current_phase["index"], int(requested_duration))

            for _ in range(int(requested_duration)):
                traci.simulationStep()
                elapsed += 1

                total_wait_now = float(get_waiting_time(self.lanes))
                active_wait_now = float(get_waiting_time(active_lanes))
                waiting_sum += total_wait_now

                if traci.simulation.getMinExpectedNumber() <= 0:
                    break

                if elapsed >= int(self.min_green) and active_wait_now <= 0.0:
                    idle_green_seconds += 1
                    break

        (
            vehicles_after,
            halts_after,
            phase_vehicle_after,
            phase_queue_after,
            phase_pressure_after,
        ) = self._current_measurements()

        active_queue_after = float(phase_queue_after[current_phase_idx])
        active_pressure_after = float(phase_pressure_after[current_phase_idx])
        other_pressure_after = np.delete(phase_pressure_after, current_phase_idx)
        other_max_after = float(other_pressure_after.max()) if other_pressure_after.size > 0 else 0.0
        total_queue_after = float(phase_queue_after.sum())

        active_queue_gain = active_queue_before - active_queue_after
        active_pressure_gain = active_pressure_before - active_pressure_after
        global_queue_gain = total_queue_before - total_queue_after
        dominance_gap_before = active_pressure_before - other_max_before
        dominance_gap_after = active_pressure_after - other_max_after
        dominance_gap_gain = dominance_gap_before - dominance_gap_after

        dominance_ratio = _safe_ratio(active_pressure_before, other_max_before + 1.0)
        dominance_weight = 1.0 + 0.65 * np.tanh(dominance_ratio - 1.0)

        # Reward intermedia:
        # - mantiene señal fuerte sobre fase activa cuando domina
        # - mantiene señal global suficiente para alinear mejor con Total waiting
        # - dominance_gap con peso moderado
        # - penalización idle moderada
        reward = (
            dominance_weight * (0.65 * active_queue_gain + 0.55 * active_pressure_gain)
            + 0.45 * global_queue_gain
            + 0.30 * dominance_gap_gain
            - 0.25 * float(idle_green_seconds)
        )

        info = {
            "active_pressure_before": active_pressure_before,
            "active_pressure_after": active_pressure_after,
            "other_max_before": other_max_before,
            "other_max_after": other_max_after,
            "dominance_ratio": float(dominance_ratio),
            "dominance_weight": float(dominance_weight),
            "active_queue_gain": float(active_queue_gain),
            "global_queue_gain": float(global_queue_gain),
            "dominance_gap_gain": float(dominance_gap_gain),
            "requested_duration": int(requested_duration),
            "elapsed_duration": int(elapsed),
            "vehicles_before": vehicles_before,
            "halts_before": halts_before,
            "vehicles_after": vehicles_after,
            "halts_after": halts_after,
            "phase_vehicle_before": phase_vehicle_before.tolist(),
            "phase_queue_before": phase_queue_before.tolist(),
            "phase_pressure_before": phase_pressure_before.tolist(),
            "phase_vehicle_after": phase_vehicle_after.tolist(),
            "phase_queue_after": phase_queue_after.tolist(),
            "phase_pressure_after": phase_pressure_after.tolist(),
        }

        self.phase_cursor = (current_phase_idx + 1) % len(self.phases)
        done = bool(traci.simulation.getMinExpectedNumber() <= 0)
        next_obs = self._obs()
        return next_obs, float(reward), done, int(elapsed), float(waiting_sum), info


def run_policy(
    *,
    episodes: int = 50,
    steps: int = 2000,
    train: bool = True,
    model_name: str = "my_policy",
    gui: bool = False,
    min_green: int = 5,
    max_green: int = 60,
) -> None:
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg"])
    try:
        junction, phases, lanes = _select_junction_phases_and_lanes(max_phases=4)
        if len(phases) <= 1:
            raise RuntimeError(
                f"Need at least 2 green phases to run the algorithm (found {len(phases)})."
            )
    finally:
        traci.close()

    model_path = os.path.join(os.path.dirname(__file__), "models", f"{model_name}.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg"])
    try:
        probe_env = SumoTrafficEnv(
            lanes,
            phases,
            junction,
            min_green=int(min_green),
            max_green=int(max_green),
        )
        probe_state = probe_env.reset()
        agent_obs_dim = int(np.asarray(probe_state, dtype=np.float32).reshape(-1).shape[0])
    finally:
        traci.close()

    agent_action_dim = 1
    agent = PPOAgent(
        agent_obs_dim,
        action_dim=agent_action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.995,
        gae_lambda=0.97,
        clip_eps=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=8,
        minibatch_size=128,
        normalize_obs=True,
    )

    if not train:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Policy not found: {model_path}. Train first with --policy-train -m {model_name}."
            )
        _ = agent.load(model_path, map_location=torch.device("cpu"))

    sim_binary = "sumo-gui" if gui else "sumo"
    debug = not train
    debug_limit = 160
    best_wait = float("inf")

    for ep in range(int(episodes)):
        traci.start(
            [
                checkBinary(sim_binary),
                "-c",
                "configuration.sumocfg",
                "--tripinfo-output",
                "maps/tripinfo.xml",
            ]
        )
        try:
            env = SumoTrafficEnv(
                lanes,
                phases,
                junction,
                min_green=int(min_green),
                max_green=int(max_green),
            )
            state = env.reset()

            total_wait = 0.0
            total_reward = 0.0
            step = 0
            debug_count = 0

            while step < int(steps) and traci.simulation.getMinExpectedNumber() > 0:
                phase_before = int(env.phase_cursor)

                raw_action01, log_prob, value, state_used = agent.act(
                    state,
                    deterministic=not train,
                    update_rms=train,
                )

                remaining_steps = int(steps) - step
                next_state, reward, done_sumo, phase_seconds, wait_sum, info = env.step(
                    raw_action01,
                    max_steps=int(remaining_steps),
                )

                if debug and debug_count < debug_limit:
                    action_dbg = np.asarray(raw_action01, dtype=np.float32)
                    print(
                        "[DEBUG] "
                        f"phase={phase_before} "
                        f"action01={np.round(action_dbg, 3).tolist()} "
                        f"req_dur={int(info.get('requested_duration', 0))} "
                        f"exec_dur={int(info.get('elapsed_duration', phase_seconds))} "
                        f"dominance={info.get('dominance_ratio', 0.0):.2f} "
                        f"active_pressure_before={info.get('active_pressure_before', 0.0):.2f} "
                        f"other_max_before={info.get('other_max_before', 0.0):.2f}"
                    )
                    debug_count += 1

                step += int(phase_seconds)
                total_wait += float(wait_sum)
                total_reward += float(reward)
                done = bool(step >= int(steps) or done_sumo)

                if train:
                    agent.buffer.add(
                        state=state_used,
                        action=np.asarray(raw_action01, dtype=np.float32),
                        log_prob=log_prob,
                        reward=reward,
                        done=done,
                        value=value,
                    )

                state = next_state
                if done:
                    break

            if train:
                update_info = agent.update(last_state=state)
                print(
                    f"Episode {ep + 1}/{episodes} | "
                    f"Total waiting: {total_wait:.0f} | "
                    f"Total reward: {total_reward:.2f} | "
                    f"policy_loss={update_info.get('policy_loss')} | "
                    f"value_loss={update_info.get('value_loss')} | "
                    f"entropy={update_info.get('entropy')}"
                )
            else:
                print(f"Total waiting: {total_wait:.0f} | Total reward: {total_reward:.2f}")

            if train and total_wait < best_wait:
                best_wait = float(total_wait)
                agent.save(
                    model_path,
                    metadata={
                        "lanes": lanes,
                        "junction": junction,
                        "phase_indices": [p["index"] for p in phases],
                        "min_green": int(min_green),
                        "max_green": int(max_green),
                        "best_wait": float(best_wait),
                        "best_episode": int(ep + 1),
                        "obs_dim": int(agent_obs_dim),
                        "action_dim": int(agent_action_dim),
                        "feature_set": "lane_vehicle_log, lane_halts_log, phase_vehicle_log, phase_queue_log, pressure_share, dominance_features",
                        "duration_mapping": "convex_pow_1_5",
                    },
                )
                print(f"New best model saved to {model_path} (best_wait={best_wait:.0f})")
        finally:
            traci.close()

    if train:
        print(f"Training finished. Best policy saved to {model_path}")