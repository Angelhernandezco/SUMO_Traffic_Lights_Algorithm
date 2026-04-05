import os
from typing import List, Tuple

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


def _build_state(vehicles_per_lane: dict, lanes: List[str]) -> np.ndarray:
    """
    Builds the observation (state) for the RL agent.
    The state is a vector of vehicle counts for each lane.
    """
    return np.array([vehicles_per_lane.get(lane, 0) for lane in lanes], dtype=np.float32)


def _build_phase_state(
    vehicles_per_lane: dict,
    lanes: List[str],
    *,
    phase_idx: int,
    num_phases: int,
) -> np.ndarray:
    """Observation = lane vehicle counts + one-hot encoding of the current phase."""

    lane_counts = _build_state(vehicles_per_lane, lanes)
    if int(num_phases) <= 0:
        raise ValueError(f"num_phases must be > 0, got {num_phases}")
    if not (0 <= int(phase_idx) < int(num_phases)):
        raise ValueError(f"phase_idx out of range: {phase_idx} (num_phases={num_phases})")

    phase_one_hot = np.zeros((int(num_phases),), dtype=np.float32)
    phase_one_hot[int(phase_idx)] = 1.0
    return np.concatenate([lane_counts, phase_one_hot], axis=0)


def _normalized_action_to_duration(action01: np.ndarray, *, min_green: int, max_green: int) -> int:
    """Converts PPO output (normalized in [0, 1]) into a green duration in seconds."""

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
    """
    Gives:
    - 1 junction (traffic light)
    - its green phases
    - all lanes controlled by those phases
    """
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
        agent_obs_dim: int,
        min_green: int,
        max_green: int,
    ) -> None:
        self.lanes = lanes
        self.phases = phases
        self.junction = junction

        self.agent_obs_dim = int(agent_obs_dim)
        self.min_green = int(min_green)
        self.max_green = int(max_green)

        self.phase_cursor = 0

        self._base_obs_dim = len(self.lanes)
        self._phase_obs_dim = len(self.lanes) + len(self.phases)

    def reset(self) -> np.ndarray:
        self.phase_cursor = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        vehicles_per_lane = get_vehicle_numbers(self.lanes)
        if int(self.agent_obs_dim) == int(self._base_obs_dim):
            return _build_state(vehicles_per_lane, self.lanes)

        return _build_phase_state(
            vehicles_per_lane,
            self.lanes,
            phase_idx=int(self.phase_cursor),
            num_phases=len(self.phases),
        )

    def step(
        self,
        action01: np.ndarray,
        *,
        agent_action_dim: int,
        max_steps: int,
    ) -> Tuple[np.ndarray, float, bool, int, float]:
        """Executes the current phase.

        Returns: next_obs, reward, done, steps_executed, waiting_sum
        """

        if int(max_steps) <= 0:
            done = bool(traci.simulation.getMinExpectedNumber() <= 0)
            return self._obs(), 0.0, done, 0, 0.0

        current_phase = self.phases[int(self.phase_cursor)]

        if int(agent_action_dim) == 1:
            action_component = action01
        else:
            action_vec = np.asarray(action01, dtype=np.float32)
            action_component = action_vec[int(self.phase_cursor)]

        duration = _normalized_action_to_duration(
            action_component,
            min_green=int(self.min_green),
            max_green=int(self.max_green),
        )
        duration = min(int(duration), int(max_steps))

        reward = 0.0
        waiting_sum = 0.0

        if duration > 0:
            set_phase_by_index(self.junction, current_phase["index"], int(duration))

            for _ in range(int(duration)):
                traci.simulationStep()
                waiting = float(get_waiting_time(self.lanes))
                waiting_sum += waiting
                reward += -waiting
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break

        self.phase_cursor = (int(self.phase_cursor) + 1) % len(self.phases)

        done = bool(traci.simulation.getMinExpectedNumber() <= 0)
        next_obs = self._obs()
        return next_obs, float(reward), done, int(duration), float(waiting_sum)


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
    """
    Runs PPO training or testing on SUMO traffic simulation.

    Core idea:
    - Observation = vehicle counts per lane
    - Action = duration for the current phase (normalized [0,1])
    - Reward = negative waiting time (we want less waiting)
    """

    # Start SUMO once to get lanes and phases
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg"])
    try:
        junction, phases, lanes = _select_junction_phases_and_lanes(max_phases=4)
        # Observation size = lane counts + one-hot phase indicator
        obs_dim = len(lanes) + len(phases)
        if len(phases) <= 1:
            raise RuntimeError(
                f"Need at least 2 green phases to run the algorithm (found {len(phases)})."
            )
    finally:
        traci.close()

    model_path = os.path.join(os.path.dirname(__file__), "models", f"{model_name}.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Training defaults to the new formulation:
    # - obs = lane counts + phase one-hot
    # - action = scalar duration for current phase
    base_obs_dim = len(lanes)
    phase_obs_dim = len(lanes) + len(phases)

    if train:
        agent_obs_dim = phase_obs_dim
        agent_action_dim = 1
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Policy not found: {model_path}. Train first with --policy-train -m {model_name}."
            )

        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        ckpt_obs_dim = None
        ckpt_action_dim = None
        if isinstance(ckpt, dict):
            if "obs_dim" in ckpt:
                ckpt_obs_dim = int(ckpt["obs_dim"])
            if "action_dim" in ckpt:
                ckpt_action_dim = int(ckpt["action_dim"])

        # If the checkpoint doesn't declare dimensions, assume legacy defaults.
        agent_obs_dim = ckpt_obs_dim if ckpt_obs_dim is not None else base_obs_dim
        agent_action_dim = ckpt_action_dim if ckpt_action_dim is not None else len(phases)

        valid_obs_dims = {base_obs_dim, phase_obs_dim}
        if int(agent_obs_dim) not in valid_obs_dims:
            raise ValueError(
                f"Checkpoint obs_dim={agent_obs_dim} is not compatible with this network. "
                f"Expected {sorted(valid_obs_dims)} (lanes={len(lanes)}, phases={len(phases)}). "
                f"Re-train with --policy-train -m {model_name}."
            )
        if int(agent_action_dim) not in {1, len(phases)}:
            raise ValueError(
                f"Checkpoint action_dim={agent_action_dim} is not compatible with this network. "
                f"Expected 1 (scalar duration) or {len(phases)} (legacy per-phase vector)."
            )

    agent = PPOAgent(agent_obs_dim, action_dim=agent_action_dim)

    if not train:
        _ = agent.load(model_path, map_location=torch.device("cpu"))

    # Choose SUMO gui or no-gui
    sim_binary = "sumo-gui" if gui else "sumo"

    debug = True

    for ep in range(int(episodes)):
        # Start a fresh simulation for each episode
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
                agent_obs_dim=int(agent_obs_dim),
                min_green=int(min_green),
                max_green=int(max_green),
            )

            state = env.reset()
            total_wait = 0.0 # total accumulated waiting time (metric)
            step = 0

            while step < int(steps) and traci.simulation.getMinExpectedNumber() > 0:
                phase_before = int(env.phase_cursor)
                raw_action01, log_prob, value = agent.act(state)

                remaining_steps = int(steps) - step
                next_state, phase_reward, done_sumo, phase_seconds, wait_sum = env.step(
                    raw_action01,
                    agent_action_dim=int(agent_action_dim),
                    max_steps=int(remaining_steps),
                )

                if debug:
                    state_dbg = np.asarray(state, dtype=np.float32)
                    action_dbg = np.asarray(raw_action01, dtype=np.float32)
                    print(
                        "[DEBUG] "
                        f"phase={phase_before} "
                        f"state={np.round(state_dbg, 3).tolist()} "
                        f"action01={np.round(action_dbg, 3).tolist()} "
                        f"duration={int(phase_seconds)}"
                    )

                step += int(phase_seconds)
                total_wait += float(wait_sum)

                done = bool(step >= int(steps) or done_sumo)

                if train:
                    agent.buffer.add(
                        state=state,
                        action=np.asarray(raw_action01, dtype=np.float32),
                        log_prob=log_prob,
                        reward=phase_reward,
                        done=done,
                        value=value,
                    )

                state = next_state

                if done:
                    break

            if train:
                agent.update(last_state=state)
                print(
                    f"Episode {ep + 1}/{episodes}"
                )
            print(
                f"Total waiting: {total_wait:.0f} "
            )

        finally:
            traci.close()

    if train:
        agent.save(
            model_path,
            metadata={
                "lanes": lanes,
                "junction": junction,
                "phase_indices": [p["index"] for p in phases],
                "min_green": int(min_green),
                "max_green": int(max_green),
            },
        )
        print(f"Policy saved to {model_path}")

