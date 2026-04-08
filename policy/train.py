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
        # Presión suave: prioriza cola, pero también ve volumen entrante.
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
    if total_pressure <= 1e-6:
        phase_share = np.zeros_like(phase_pressure, dtype=np.float32)
    else:
        phase_share = phase_pressure / total_pressure

    order = _cyclic_order(int(phase_idx), num_phases)
    current = order[0]
    next_phase = order[1]
    rest = order[2:]

    def phase_feat(i: int, include_relative: bool) -> np.ndarray:
        base = np.asarray(
            [
                np.log1p(float(phase_vehicle[i])),
                np.log1p(float(phase_queue[i])),
                np.log1p(float(phase_pressure[i])),
                float(phase_share[i]),
            ],
            dtype=np.float32,
        )
        if not include_relative:
            return base

        gap_next = (float(phase_pressure[i]) - float(phase_pressure[next_phase])) / (total_pressure + 1.0)
        others = [j for j in range(num_phases) if j != i]
        max_other = max(float(phase_pressure[j]) for j in others) if others else 0.0
        gap_max_other = (float(phase_pressure[i]) - max_other) / (total_pressure + 1.0)
        rel = np.asarray([gap_next, gap_max_other], dtype=np.float32)
        return np.concatenate([base, rel], axis=0)

    current_feat = phase_feat(current, include_relative=True)
    next_feat = phase_feat(next_phase, include_relative=True)

    rest_feats: List[np.ndarray] = []
    for i in rest:
        rest_feats.append(phase_feat(i, include_relative=False))
    rest_flat = np.concatenate(rest_feats, axis=0) if rest_feats else np.zeros((0,), dtype=np.float32)

    max_other_pressure = max(float(phase_pressure[j]) for j in order[1:]) if order[1:] else 0.0
    max_other_share = max(float(phase_share[j]) for j in order[1:]) if order[1:] else 0.0
    summary = np.asarray(
        [
            np.log1p(total_pressure),
            float(phase_share[current]),
            float(phase_share[next_phase]),
            float(max_other_share),
            (float(phase_pressure[current]) - float(phase_pressure[next_phase])) / (total_pressure + 1.0),
            (float(phase_pressure[current]) - max_other_pressure) / (total_pressure + 1.0),
        ],
        dtype=np.float32,
    )

    return np.concatenate([current_feat, next_feat, rest_flat, summary], axis=0)


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
    current = order[0]
    next_phase = order[1]
    rest = order[2:]

    total_pressure = float(np.sum(phase_pressure))
    shares = phase_pressure / max(total_pressure, 1.0)
    current_pressure = float(phase_pressure[current])
    next_pressure = float(phase_pressure[next_phase])
    max_other_pressure = max(float(phase_pressure[j]) for j in order[1:]) if order[1:] else 0.0
    other_total = sum(float(phase_pressure[j]) for j in order[1:]) if order[1:] else 0.0

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
        "current_pressure": current_pressure,
        "next_pressure": next_pressure,
        "max_other_pressure": max_other_pressure,
        "other_total_pressure": float(other_total),
        "current_share": float(shares[current]),
        "next_share": float(shares[next_phase]),
        "max_other_share": max(float(shares[j]) for j in order[1:]) if order[1:] else 0.0,
        "total_pressure": total_pressure,
        "phase_idx": int(phase_idx),
        "current_idx": int(current),
        "next_idx": int(next_phase),
        "rest_idx": np.asarray(rest, dtype=np.int64),
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
        snap = self._snapshot()
        return np.asarray(snap["state"], dtype=np.float32)

    def step(self, action01: np.ndarray, *, max_steps: int) -> Tuple[np.ndarray, float, bool, int, float, dict]:
        if int(max_steps) <= 0:
            done = bool(traci.simulation.getMinExpectedNumber() <= 0)
            return self._obs(), 0.0, done, 0, 0.0, {
                "requested_duration": 0,
                "executed_duration": 0,
            }

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
        before_next = float(before["next_pressure"])
        before_total = float(before["total_pressure"])
        before_share = float(before["current_share"])
        before_next_share = float(before["next_share"])
        before_max_other = float(before["max_other_pressure"])
        after_curr = float(after_same_phase["current_pressure"])
        after_total = float(after_same_phase["total_pressure"])

        served_current = np.clip((before_curr - after_curr) / (before_curr + 1.0), -1.0, 1.0)
        global_relief = np.clip((before_total - after_total) / (before_total + 1.0), -1.0, 1.0)

        extra_green = max(0, int(executed_duration) - int(self.min_green))
        green_span = max(1, int(self.max_green) - int(self.min_green))
        extra_green_ratio = float(extra_green) / float(green_span)

        dominance_soft = max(
            0.0,
            (before_curr - before_max_other) / (before_total + 1.0),
        )
        next_stronger_soft = max(
            0.0,
            (before_next - before_curr) / (before_total + 1.0),
        )
        low_demand_soft = np.clip((0.18 - before_share) / 0.18, 0.0, 1.0)
        no_queue_soft = np.clip(
            1.0 - float(before["phase_queue"][current_phase_idx]) / 1.5,
            0.0,
            1.0,
        )
        waste_soft = max(low_demand_soft, no_queue_soft)

        reward = (
            -mean_wait
            + 5.0 * served_current
            + 2.5 * global_relief
            + 4.0 * dominance_soft * served_current * extra_green_ratio
            - 2.5 * waste_soft * extra_green_ratio
            - 2.0 * next_stronger_soft * extra_green_ratio
        )

        self.phase_cursor = (current_phase_idx + 1) % len(self.phases)
        done = bool(traci.simulation.getMinExpectedNumber() <= 0)
        next_obs = self._obs()

        info = {
            "requested_duration": int(requested_duration),
            "executed_duration": int(executed_duration),
            "mean_wait": float(mean_wait),
            "current_pressure": float(before_curr),
            "next_pressure": float(before_next),
            "current_share": float(before_share),
            "next_share": float(before_next_share),
            "max_other_pressure": float(before_max_other),
            "served_current": float(served_current),
            "global_relief": float(global_relief),
            "extra_green_ratio": float(extra_green_ratio),
            "dominance_soft": float(dominance_soft),
            "waste_soft": float(waste_soft),
            "next_stronger_soft": float(next_stronger_soft),
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
                f"share={info.get('current_share', 0.0):.3f} "
                f"action01={np.round(action_dbg, 3).tolist()} "
                f"req_dur={int(info.get('requested_duration', 0))} "
                f"exec_dur={int(info.get('executed_duration', phase_seconds))} "
                f"served={info.get('served_current', 0.0):.3f} "
                f"waste={info.get('waste_soft', 0.0):.3f} "
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
        if len(phases) <= 1:
            raise RuntimeError(
                f"Need at least 2 green phases to run the algorithm (found {len(phases)})."
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
        entropy_coef=0.02,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=10,
        minibatch_size=64,
        normalize_obs=True,
        concentration_floor=1.0,
        init_concentration=2.0,
    )

    metadata = {}
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

    for ep in range(int(episodes)):
        if train:
            traci.start(
                [
                    checkBinary("sumo"),
                    "-c",
                    "configuration.sumocfg",
                    "--tripinfo-output",
                    "maps/tripinfo.xml",
                ]
            )
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

            traci.start(
                [
                    checkBinary("sumo"),
                    "-c",
                    "configuration.sumocfg",
                    "--tripinfo-output",
                    "maps/tripinfo.xml",
                ]
            )
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
                f"entropy={update_info.get('entropy')}"
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
                        "state_version": "current_next_rest_structured_v1",
                        "reward_version": "global_wait_plus_useful_extra_green_v1",
                    },
                )
                print(
                    f"New best deterministic model saved to {model_path} "
                    f"(best_eval_wait={best_eval_wait:.0f})"
                )
        else:
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
