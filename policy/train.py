import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import traci
from sumolib import checkBinary

from sumo_utils import get_green_phases, set_phase_by_index
from policy.agent import PPOAgent


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
    duration = int(min_green) + int(np.rint(a01 * span))
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


def _build_phase_lane_indices(phases: list, lanes: List[str]) -> List[np.ndarray]:
    lane_to_idx = {lane: i for i, lane in enumerate(lanes)}
    phase_lane_indices: List[np.ndarray] = []
    for phase in phases:
        idxs = sorted({lane_to_idx[lane] for lane in phase["lanes"] if lane in lane_to_idx})
        if not idxs:
            raise RuntimeError(f"Phase {phase['index']} has no mapped lanes.")
        phase_lane_indices.append(np.asarray(idxs, dtype=np.int64))
    return phase_lane_indices


def _lane_snapshot(lanes: List[str]) -> Dict[str, np.ndarray]:
    lane_vehicle = np.asarray(
        [float(traci.lane.getLastStepVehicleNumber(lane)) for lane in lanes],
        dtype=np.float32,
    )
    lane_queue = np.asarray(
        [float(traci.lane.getLastStepHaltingNumber(lane)) for lane in lanes],
        dtype=np.float32,
    )
    return {
        "lane_vehicle": lane_vehicle,
        "lane_queue": lane_queue,
    }


def _phase_vectors(
    lane_vehicle: np.ndarray,
    lane_queue: np.ndarray,
    phase_lane_indices: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_phases = len(phase_lane_indices)
    phase_vehicle = np.zeros((num_phases,), dtype=np.float32)
    phase_queue = np.zeros((num_phases,), dtype=np.float32)
    phase_pressure = np.zeros((num_phases,), dtype=np.float32)

    for i, idxs in enumerate(phase_lane_indices):
        veh = float(np.sum(lane_vehicle[idxs]))
        que = float(np.sum(lane_queue[idxs]))
        prs = que + 0.35 * veh
        phase_vehicle[i] = veh
        phase_queue[i] = que
        phase_pressure[i] = prs

    return phase_vehicle, phase_queue, phase_pressure


def _cyclic_order(current: int, num_phases: int) -> List[int]:
    return [int((current + k) % num_phases) for k in range(num_phases)]


def _state_from_phase_stats(
    phase_vehicle: np.ndarray,
    phase_queue: np.ndarray,
    phase_pressure: np.ndarray,
    *,
    phase_idx: int,
) -> np.ndarray:
    num_phases = int(phase_vehicle.shape[0])
    if num_phases <= 1:
        raise ValueError(f"num_phases must be > 1, got {num_phases}")
    if not (0 <= int(phase_idx) < num_phases):
        raise ValueError(f"phase_idx out of range: {phase_idx} (num_phases={num_phases})")

    total_pressure = float(np.sum(phase_pressure))
    denom = total_pressure + 1.0
    phase_share = phase_pressure / denom

    order = _cyclic_order(int(phase_idx), num_phases)

    def phase_block(i: int) -> np.ndarray:
        return np.asarray(
            [
                np.log1p(float(phase_vehicle[i])),
                np.log1p(float(phase_queue[i])),
                np.log1p(float(phase_pressure[i])),
                float(phase_share[i]),
            ],
            dtype=np.float32,
        )

    blocks = [phase_block(i) for i in order]  # [current, next, next+1, next+2]

    future_pressures = [float(phase_pressure[i]) for i in order[1:]]
    weighted_future_pressure = (
        1.00 * future_pressures[0]
        + 0.70 * future_pressures[1]
        + 0.40 * future_pressures[2]
    )
    peak_future_pos = int(np.argmax(future_pressures)) + 1
    max_future_pressure = max(future_pressures)
    sum_future_pressure = sum(future_pressures)
    current_pressure = float(phase_pressure[order[0]])
    gap_current_vs_peak_future = (current_pressure - max_future_pressure) / denom
    current_vs_next_gap = (current_pressure - float(phase_pressure[order[1]])) / denom
    current_vs_next2_gap = (current_pressure - float(phase_pressure[order[2]])) / denom
    current_vs_next3_gap = (current_pressure - float(phase_pressure[order[3]])) / denom

    summary = np.asarray(
        [
            np.log1p(total_pressure),
            np.log1p(weighted_future_pressure),
            np.log1p(max_future_pressure),
            np.log1p(sum_future_pressure),
            float(gap_current_vs_peak_future),
            float(current_vs_next_gap),
            float(current_vs_next2_gap),
            float(current_vs_next3_gap),
            float(peak_future_pos / 3.0),
        ],
        dtype=np.float32,
    )

    return np.concatenate(blocks + [summary], axis=0)


def _structured_snapshot(
    lanes: List[str],
    phase_lane_indices: List[np.ndarray],
    *,
    phase_idx: int,
) -> Dict[str, np.ndarray | float | int]:
    lane = _lane_snapshot(lanes)
    phase_vehicle, phase_queue, phase_pressure = _phase_vectors(
        lane["lane_vehicle"],
        lane["lane_queue"],
        phase_lane_indices,
    )

    num_phases = len(phase_lane_indices)
    order = _cyclic_order(int(phase_idx), num_phases)
    total_pressure = float(np.sum(phase_pressure))
    denom = total_pressure + 1.0
    shares = phase_pressure / denom

    future_pressures = [float(phase_pressure[i]) for i in order[1:]]
    weighted_future_pressure = (
        1.00 * future_pressures[0]
        + 0.70 * future_pressures[1]
        + 0.40 * future_pressures[2]
    )
    peak_future_pos = int(np.argmax(future_pressures)) + 1
    peak_future_idx = int(order[peak_future_pos])
    max_future_pressure = max(future_pressures)
    sum_future_pressure = sum(future_pressures)
    current_pressure = float(phase_pressure[order[0]])
    next_pressure = float(phase_pressure[order[1]])
    next2_pressure = float(phase_pressure[order[2]])
    next3_pressure = float(phase_pressure[order[3]])

    state = _state_from_phase_stats(
        phase_vehicle,
        phase_queue,
        phase_pressure,
        phase_idx=int(phase_idx),
    )

    return {
        "state": state,
        "phase_vehicle": phase_vehicle,
        "phase_queue": phase_queue,
        "phase_pressure": phase_pressure,
        "phase_share": shares,
        "order": np.asarray(order, dtype=np.int64),
        "current_pressure": current_pressure,
        "next_pressure": next_pressure,
        "next2_pressure": next2_pressure,
        "next3_pressure": next3_pressure,
        "current_share": float(shares[order[0]]),
        "next_share": float(shares[order[1]]),
        "next2_share": float(shares[order[2]]),
        "next3_share": float(shares[order[3]]),
        "total_pressure": total_pressure,
        "weighted_future_pressure": float(weighted_future_pressure),
        "max_future_pressure": float(max_future_pressure),
        "sum_future_pressure": float(sum_future_pressure),
        "peak_future_pos": int(peak_future_pos),
        "peak_future_idx": int(peak_future_idx),
        "phase_idx": int(phase_idx),
    }


class SumoTrafficEnv:
    def __init__(
        self,
        lanes: List[str],
        phases: list,
        phase_lane_indices: List[np.ndarray],
        junction: str,
        *,
        min_green: int,
        max_green: int,
    ) -> None:
        self.lanes = lanes
        self.phases = phases
        self.phase_lane_indices = phase_lane_indices
        self.junction = junction
        self.min_green = int(min_green)
        self.max_green = int(max_green)
        self.phase_cursor = 0

    def reset(self) -> np.ndarray:
        self.phase_cursor = 0
        return self._obs()

    def _snapshot(self, phase_idx: int | None = None) -> Dict[str, np.ndarray | float | int]:
        idx = int(self.phase_cursor if phase_idx is None else phase_idx)
        return _structured_snapshot(self.lanes, self.phase_lane_indices, phase_idx=idx)

    def _obs(self) -> np.ndarray:
        return np.asarray(self._snapshot()["state"], dtype=np.float32)

    def step(self, action01: np.ndarray, *, max_steps: int) -> Tuple[np.ndarray, float, bool, int, float, dict]:
        if int(max_steps) <= 0:
            done = bool(traci.simulation.getMinExpectedNumber() <= 0)
            return self._obs(), 0.0, done, 0, 0.0, {"requested_duration": 0, "executed_duration": 0}

        current_phase_idx = int(self.phase_cursor)
        current_phase = self.phases[current_phase_idx]
        before = self._snapshot(current_phase_idx)

        requested_duration = _normalized_action_to_duration(
            action01,
            min_green=int(self.min_green),
            max_green=int(self.max_green),
        )
        requested_duration = min(int(requested_duration), int(max_steps))

        waiting_sum = 0.0
        executed_duration = 0

        if requested_duration > 0:
            set_phase_by_index(self.junction, current_phase["index"], int(requested_duration))
            for _ in range(int(requested_duration)):
                traci.simulationStep()
                executed_duration += 1
                waiting_sum += float(sum(traci.lane.getLastStepHaltingNumber(l) for l in self.lanes))
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break

        after_same_phase = self._snapshot(current_phase_idx)

        mean_wait = waiting_sum / max(1, executed_duration)
        before_curr = float(before["current_pressure"])
        before_total = float(before["total_pressure"])
        before_current_share = float(before["current_share"])
        before_peak_future = float(before["max_future_pressure"])
        before_weighted_future = float(before["weighted_future_pressure"])
        before_peak_future_pos = int(before["peak_future_pos"])
        after_curr = float(after_same_phase["current_pressure"])
        after_total = float(after_same_phase["total_pressure"])

        served_current = np.clip((before_curr - after_curr) / (before_curr + 1.0), -1.0, 1.0)
        global_relief = np.clip((before_total - after_total) / (before_total + 1.0), -1.0, 1.0)

        extra_green = max(0, int(executed_duration) - int(self.min_green))
        green_span = max(1, int(self.max_green) - int(self.min_green))
        extra_green_ratio = float(extra_green) / float(green_span)
        extra_green_sq = float(extra_green_ratio * extra_green_ratio)

        denom = before_total + 1.0
        dominance_soft = max(0.0, (before_curr - before_peak_future) / denom)
        future_urgency_soft = min(1.0, before_weighted_future / denom)
        future_peak_gap_soft = max(0.0, (before_peak_future - before_curr) / denom)
        low_pressure_soft = np.clip(1.0 - before_curr / 8.0, 0.0, 1.0)
        low_next_soft = np.clip(1.0 - float(before["next_pressure"]) / 8.0, 0.0, 1.0)
        low_next2_soft = np.clip(1.0 - float(before["next2_pressure"]) / 8.0, 0.0, 1.0)
        weak_current_soft = max(low_pressure_soft, 1.0 - min(1.0, before_current_share / 0.25))
        queue_now = float(before["phase_queue"][current_phase_idx])
        no_queue_soft = np.clip(1.0 - queue_now / 1.5, 0.0, 1.0)
        waste_soft = max(weak_current_soft, no_queue_soft)

        # Jerarquía buscada:
        # 1) Servir bien una fase dominante.
        # 2) No dejar corta una fase que realmente domina.
        # 3) Fast-pass solo cuando current es débil y sí hay valor en llegar antes
        #    a una fase más fuerte. Cuando todo está vacío, fast-pass existe pero
        #    con mucho menos peso.
        chain_empty_soft = np.clip(
            low_pressure_soft * (0.65 + 0.35 * low_next_soft) * (0.75 + 0.25 * low_next2_soft),
            0.0,
            1.5,
        )

        distance_weight = float(before_peak_future_pos / 3.0)  # 1/3, 2/3 o 1.0
        dominance_gate = max(before_current_share, dominance_soft)

        # Valor real de "apurar": cuánto más fuerte se ve lo que viene frente a current.
        future_pull_soft = np.clip(
            0.70 * future_peak_gap_soft + 0.45 * future_urgency_soft,
            0.0,
            1.0,
        )

        # Cuando literalmente casi todo está bajo, fast-pass existe pero vale poco.
        dead_cycle_soft = np.clip(
            low_pressure_soft * low_next_soft * low_next2_soft * (1.0 - future_pull_soft),
            0.0,
            1.0,
        )

        fast_pass_context_soft = np.clip(
            chain_empty_soft
            * low_pressure_soft
            * (0.20 * dead_cycle_soft + 1.00 * future_pull_soft * (0.55 + 0.45 * distance_weight))
            * (1.0 - 0.75 * dominance_gate),
            0.0,
            1.5,
        )

        # Costo por quedarse verde en current mientras lo fuerte está adelante.
        bad_delay_cost = extra_green_sq * future_pull_soft * (0.75 + 0.95 * distance_weight)

        # Penalización por extensión inútil: más baja si current sí domina y está sirviendo.
        dominance_relief = 1.0 - 0.40 * dominance_gate * max(0.0, served_current)
        bad_extension_soft = extra_green_sq * dominance_relief * (
            1.00 * waste_soft
            + 1.40 * future_peak_gap_soft
            + 0.55 * future_urgency_soft
            + 0.30 * (1.0 - before_current_share)
        )

        dominant_extension_soft = extra_green_ratio * max(0.0, served_current) * (
            1.10 * dominance_soft
            + 1.00 * before_current_share
            + 0.45 * min(1.0, before_curr / 12.0)
        )

        dominant_need_soft = min(1.0, before_curr / 10.0) * dominance_gate
        under_green_soft = (1.0 - extra_green_ratio) * dominant_need_soft * (0.70 + 0.90 * dominance_gate)

        reward = (
            -mean_wait
            + 5.0 * served_current
            + 1.5 * global_relief
            + 5.5 * fast_pass_context_soft
            + 20.0 * dominant_extension_soft
            - 4.0 * bad_extension_soft
            - 8.5 * bad_delay_cost
            - 1.0 * extra_green_ratio * waste_soft
            - 8.5 * under_green_soft
        )

        self.phase_cursor = (current_phase_idx + 1) % len(self.phases)
        done = bool(traci.simulation.getMinExpectedNumber() <= 0)
        next_obs = self._obs()

        info = {
            "requested_duration": int(requested_duration),
            "executed_duration": int(executed_duration),
            "mean_wait": float(mean_wait),
            "current_pressure": float(before_curr),
            "next_pressure": float(before["next_pressure"]),
            "next2_pressure": float(before["next2_pressure"]),
            "next3_pressure": float(before["next3_pressure"]),
            "current_share": float(before_current_share),
            "peak_future_pos": int(before_peak_future_pos),
            "weighted_future_pressure": float(before_weighted_future),
            "served_current": float(served_current),
            "global_relief": float(global_relief),
            "extra_green_ratio": float(extra_green_ratio),
            "dominance_soft": float(dominance_soft),
            "waste_soft": float(waste_soft),
            "bad_delay_cost": float(bad_delay_cost),
            "bad_extension_soft": float(bad_extension_soft),
            "fast_pass_bonus": float(fast_pass_context_soft),
            "future_pull_soft": float(future_pull_soft),
            "dominance_gate": float(dominance_gate),
            "dominant_extension_soft": float(dominant_extension_soft),
            "dominant_need_soft": float(dominant_need_soft),
            "under_green_soft": float(under_green_soft),
            "reward": float(reward),
        }
        return next_obs, float(reward), done, int(executed_duration), float(waiting_sum), info


def _run_single_episode(
    *,
    agent: PPOAgent,
    env: SumoTrafficEnv,
    steps: int,
    collect_rollout: bool,
    deterministic: bool,
    debug: bool,
    debug_limit: int,
) -> dict:
    state = env.reset()
    total_wait = 0.0
    total_reward = 0.0
    step = 0
    debug_count = 0

    while step < int(steps) and traci.simulation.getMinExpectedNumber() > 0:
        phase_before = int(env.phase_cursor)

        raw_action01, log_prob, value, prepared_state = agent.act(
            state,
            deterministic=deterministic,
            update_rms=collect_rollout,
        )

        remaining_steps = int(steps) - step
        next_state, phase_reward, done_sumo, phase_seconds, wait_sum, info = env.step(
            raw_action01,
            max_steps=int(remaining_steps),
        )

        if debug and debug_count < debug_limit:
            action_dbg = np.asarray(raw_action01, dtype=np.float32)
            print(
                "[DEBUG] "
                f"phase={phase_before} "
                f"curr_p={info.get('current_pressure', 0.0):.2f} "
                f"next_p={info.get('next_pressure', 0.0):.2f} "
                f"next2_p={info.get('next2_pressure', 0.0):.2f} "
                f"next3_p={info.get('next3_pressure', 0.0):.2f} "
                f"share={info.get('current_share', 0.0):.3f} "
                f"peak_pos={int(info.get('peak_future_pos', 0))} "
                f"action01={np.round(action_dbg, 3).tolist()} "
                f"req_dur={int(info.get('requested_duration', 0))} "
                f"exec_dur={int(info.get('executed_duration', phase_seconds))} "
                f"served={info.get('served_current', 0.0):.3f} "
                f"waste={info.get('waste_soft', 0.0):.3f} "
                f"bad_delay={info.get('bad_delay_cost', 0.0):.3f} "
                f"fast_pass={info.get('fast_pass_bonus', 0.0):.3f} "
                f"bad_ext={info.get('bad_extension_soft', 0.0):.3f} "
                f"under_g={info.get('under_green_soft', 0.0):.3f} "
                f"dom={info.get('dominance_soft', 0.0):.3f} "
                f"r={info.get('reward', 0.0):.3f}"
            )
            debug_count += 1

        step += int(phase_seconds)
        total_wait += float(wait_sum)
        total_reward += float(phase_reward)
        done = bool(step >= int(steps) or done_sumo)

        if collect_rollout:
            agent.buffer.add(
                state=prepared_state,
                action=np.asarray(raw_action01, dtype=np.float32),
                log_prob=log_prob,
                reward=phase_reward,
                done=done,
                value=value,
            )

        state = next_state
        if done:
            break

    return {
        "last_state": state,
        "total_wait": float(total_wait),
        "total_reward": float(total_reward),
        "steps": int(step),
    }


def _make_env(*, lanes, phases, phase_lane_indices, junction, min_green, max_green) -> SumoTrafficEnv:
    return SumoTrafficEnv(
        lanes=lanes,
        phases=phases,
        phase_lane_indices=phase_lane_indices,
        junction=junction,
        min_green=int(min_green),
        max_green=int(max_green),
    )


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
        if len(phases) != 4:
            raise RuntimeError(
                f"This version expects exactly 4 green phases (found {len(phases)})."
            )
        phase_lane_indices = _build_phase_lane_indices(phases, lanes)
        sample_obs = _structured_snapshot(lanes, phase_lane_indices, phase_idx=0)["state"]
        obs_dim = int(np.asarray(sample_obs, dtype=np.float32).shape[0])
    finally:
        traci.close()

    model_path = os.path.join(os.path.dirname(__file__), "models", f"{model_name}.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    agent = PPOAgent(
        obs_dim,
        action_dim=1,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.07,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=10,
        minibatch_size=64,
        normalize_obs=True,
        concentration_floor=0.2,
        init_action_mean=0.08,
        init_total_concentration=2.8,
    )

    if not train:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Policy not found: {model_path}. Train first with --policy-train -m {model_name}."
            )
        metadata = agent.load(model_path, map_location=torch.device("cpu"))
        saved_min = metadata.get("min_green")
        saved_max = metadata.get("max_green")
        if saved_min is not None and int(saved_min) != int(min_green):
            raise ValueError(
                f"This model was trained with min_green={saved_min}, but test is using min_green={min_green}. "
                "Use the same min_green in train and test."
            )
        if saved_max is not None and int(saved_max) != int(max_green):
            raise ValueError(
                f"This model was trained with max_green={saved_max}, but test is using max_green={max_green}. "
                "Use the same max_green in train and test."
            )

    sim_binary = "sumo-gui" if gui else "sumo"
    debug = not train
    debug_limit = 250
    best_eval_wait = float("inf")

    entropy_start = 0.07
    entropy_end = 0.010
    entropy_decay_portion = 0.55

    for ep in range(int(episodes)):
        if train:
            if int(episodes) <= 1:
                entropy_coef_now = entropy_end
            else:
                progress = min(1.0, float(ep) / max(1.0, entropy_decay_portion * (int(episodes) - 1)))
                entropy_coef_now = entropy_start + (entropy_end - entropy_start) * progress
            agent.set_entropy_coef(entropy_coef_now)

            traci.start([
                checkBinary("sumo"),
                "-c", "configuration.sumocfg",
                "--tripinfo-output", "maps/tripinfo.xml",
            ])
            try:
                train_env = _make_env(
                    lanes=lanes,
                    phases=phases,
                    phase_lane_indices=phase_lane_indices,
                    junction=junction,
                    min_green=min_green,
                    max_green=max_green,
                )
                train_metrics = _run_single_episode(
                    agent=agent,
                    env=train_env,
                    steps=int(steps),
                    collect_rollout=True,
                    deterministic=False,
                    debug=False,
                    debug_limit=0,
                )
            finally:
                traci.close()

            update_info = agent.update(last_state=train_metrics["last_state"])

            traci.start([
                checkBinary("sumo"),
                "-c", "configuration.sumocfg",
                "--tripinfo-output", "maps/tripinfo.xml",
            ])
            try:
                eval_env = _make_env(
                    lanes=lanes,
                    phases=phases,
                    phase_lane_indices=phase_lane_indices,
                    junction=junction,
                    min_green=min_green,
                    max_green=max_green,
                )
                eval_metrics = _run_single_episode(
                    agent=agent,
                    env=eval_env,
                    steps=int(steps),
                    collect_rollout=False,
                    deterministic=True,
                    debug=False,
                    debug_limit=0,
                )
            finally:
                traci.close()

            print(
                f"Episode {ep + 1}/{episodes} | "
                f"Train waiting: {train_metrics['total_wait']:.0f} | "
                f"Train reward: {train_metrics['total_reward']:.2f} | "
                f"Eval waiting(det): {eval_metrics['total_wait']:.0f} | "
                f"Eval reward(det): {eval_metrics['total_reward']:.2f} | "
                f"policy_loss={update_info.get('policy_loss')} | "
                f"value_loss={update_info.get('value_loss')} | "
                f"entropy={update_info.get('entropy')} | "
                f"entropy_coef={agent.entropy_coef:.4f}"
            )

            if float(eval_metrics["total_wait"]) < best_eval_wait:
                best_eval_wait = float(eval_metrics["total_wait"])
                agent.save(
                    model_path,
                    metadata={
                        "lanes": lanes,
                        "junction": junction,
                        "phase_indices": [p["index"] for p in phases],
                        "min_green": int(min_green),
                        "max_green": int(max_green),
                        "best_eval_wait": float(best_eval_wait),
                        "best_episode": int(ep + 1),
                        "obs_dim": int(obs_dim),
                        "action_dim": 1,
                        "normalize_obs": True,
                        "state_version": "current_next_next2_next3_with_future_pressure_v1",
                        "reward_version": "global_wait_future_delay_cost_v3_extreme_push",
                        "init_action_mean": 0.08,
                        "init_total_concentration": 9.0,
                        "entropy_start": entropy_start,
                        "entropy_end": entropy_end,
                        "init_total_concentration": 2.8,
                        "concentration_floor": 0.2,
                    },
                )
                print(
                    f"New best deterministic model saved to {model_path} "
                    f"(best_eval_wait={best_eval_wait:.0f})"
                )
        else:
            traci.start([
                checkBinary(sim_binary),
                "-c", "configuration.sumocfg",
                "--tripinfo-output", "maps/tripinfo.xml",
            ])
            try:
                env = _make_env(
                    lanes=lanes,
                    phases=phases,
                    phase_lane_indices=phase_lane_indices,
                    junction=junction,
                    min_green=min_green,
                    max_green=max_green,
                )
                test_metrics = _run_single_episode(
                    agent=agent,
                    env=env,
                    steps=int(steps),
                    collect_rollout=False,
                    deterministic=True,
                    debug=debug,
                    debug_limit=debug_limit,
                )
            finally:
                traci.close()

            print(
                f"Total waiting: {test_metrics['total_wait']:.0f} | "
                f"Total reward: {test_metrics['total_reward']:.2f}"
            )

    if train:
        print(f"Best deterministic policy saved to {model_path}")
