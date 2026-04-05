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


def _normalized_actions_to_phase_durations(
    actions01: np.ndarray, *, min_green: int, max_green: int
) -> List[int]:
    """
    Converts PPO output (normalized actions in [0, 1])
    into real green light durations in seconds.
    """
    actions01 = np.asarray(actions01, dtype=np.float32)
    if actions01.ndim != 1:
        raise ValueError("actions01 must be a 1D array")
    if int(max_green) < int(min_green):
        raise ValueError(
            f"max_green={max_green} must be >= min_green={min_green}"
        )

    # Ensures the value is between 0 and 1
    actions01 = np.clip(actions01, 0.0, 1.0)
    # Convert normalized actions (values between 0 and 1) into real durations in seconds
    span = int(max_green) - int(min_green)
    durations = int(min_green) + np.rint(actions01 * span).astype(int)
    durations = np.maximum(durations, int(min_green))
    return [int(d) for d in durations.tolist()]


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
    - Action = green light duration per phase (normalized [0,1])
    - Reward = negative waiting time (we want less waiting)
    """

    # Start SUMO once to get lanes and phases
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg"])
    try:
        junction, phases, lanes = _select_junction_phases_and_lanes(max_phases=4)
        # Observation size = number of lanes
        obs_dim = len(lanes)
        # Action size = number of traffic phases
        action_dim = len(phases)
        if action_dim <= 1:
            raise RuntimeError(
                f"Need at least 2 green phases to run the algorithm (found {action_dim})."
            )
    finally:
        traci.close()

    agent = PPOAgent(obs_dim, action_dim)
    model_path = os.path.join(os.path.dirname(__file__), "models", f"{model_name}.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # If testing mode, load pretrained model
    if not train:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Policy not found: {model_path}. Train first with --policy-train -m {model_name}."
            )
        _ = agent.load(model_path, map_location=torch.device("cpu"))

    # Choose SUMO gui or no-gui
    sim_binary = "sumo-gui" if gui else "sumo"

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
            state = _build_state(get_vehicle_numbers(lanes), lanes)
            total_wait = 0.0 # total accumulated waiting time (metric)
            step = 0

            while step < int(steps) and traci.simulation.getMinExpectedNumber() > 0:
                actions01, log_prob, value = agent.act(state)
                durations = _normalized_actions_to_phase_durations(
                    actions01,
                    min_green=int(min_green),
                    max_green=int(max_green)
                )

                #print("\n[DEBUG] Phase durations (seconds assigned):")
                #for i, (phase, d) in enumerate(zip(phases, durations)):
                #    print(f"  Phase {i} (index {phase['index']}): {d} sec")

                #print(f"  Raw actions01: {np.round(actions01, 3)}\n")

                cycle_reward = 0.0

                # Execute each traffic light phase
                for phase_idx, phase in enumerate(phases):
                    phase_seconds = int(durations[phase_idx])

                    # skip if no time assigned
                    if phase_seconds <= 0:
                        continue

                    # ensure we don't exceed episode length
                    remaining_steps = int(steps) - step
                    if remaining_steps <= 0:
                        break

                    phase_seconds = min(phase_seconds, remaining_steps)

                    # Set traffic light phase in SUMO
                    set_phase_by_index(junction, phase["index"], phase_seconds)

                    for _ in range(phase_seconds):
                        traci.simulationStep() # advance simulation
                        waiting = float(get_waiting_time(lanes))
                        total_wait += waiting
                        cycle_reward += -waiting
                        step += 1
                        if step >= int(steps) or traci.simulation.getMinExpectedNumber() <= 0:
                            break

                    if step >= int(steps) or traci.simulation.getMinExpectedNumber() <= 0:
                        break

                next_state = _build_state(get_vehicle_numbers(lanes), lanes)
                done = bool(
                    step >= int(steps) or traci.simulation.getMinExpectedNumber() <= 0
                )

                if train:
                    agent.buffer.add(
                        state=state,
                        action=actions01,
                        log_prob=log_prob,
                        reward=cycle_reward,
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

