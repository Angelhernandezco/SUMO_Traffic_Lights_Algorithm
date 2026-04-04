import os
from typing import List

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

from .agent import PolicyGradientAgent


def build_phase_state(vehicles_per_lane, phases):
    """State vector of size 4: vehicles per green phase.

    This aligns the observation directly with the action space (4 phase durations).
    """

    per_phase = []
    for phase in phases:
        per_phase.append(sum(float(vehicles_per_lane.get(lane, 0)) for lane in phase["lanes"]))
    return np.asarray(per_phase, dtype=np.float32)


def _compute_returns(rewards: List[float], gamma: float) -> List[float]:
    returns: List[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        returns.insert(0, running)
    return returns


def run_policy(
    *,
    episodes: int = 50,
    steps: int = 500,
    train: bool = True,
    model_name: str = "policy_model",
    gui: bool = False,
    cycle_time: int = 120,
    min_green: int = 5,
):
    """Policy-gradient control that allocates a fixed cycle_time among 4 green phases.

    Assumption (per user spec): the controlled junction has exactly 4 green phases.
    Each decision assigns durations that sum to cycle_time.
    """

    # Start SUMO once to discover lanes/phases.
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg"])
    junction = traci.trafficlight.getIDList()[0]
    phases = get_green_phases(junction)

    if len(phases) != 4:
        traci.close()
        raise ValueError(
            f"Policy timing optimizer expects exactly 4 green phases, but found {len(phases)}. "
            "If your map has a different number of green phases, adjust the algorithm or map."
        )

    lanes = sorted(set(lane for phase in phases for lane in phase["lanes"]))
    # Observation is 4 numbers (vehicles per phase), not per-lane.
    state_size = 4
    action_size = 4

    agent = PolicyGradientAgent(state_size, action_size)
    model_path = os.path.join(os.path.dirname(__file__), "models", f"{model_name}.pth")

    if not train:
        if not os.path.exists(model_path):
            traci.close()
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train first with --policy-train -m {model_name}."
            )
        agent.load(model_path, map_location=torch.device("cpu"))
        print(f"Loaded policy model from {model_path}")

    traci.close()

    sim_binary = "sumo-gui" if gui else "sumo"

    for episode in range(episodes):
        traci.start([checkBinary(sim_binary), "-c", "configuration.sumocfg"])

        log_probs = []
        values = []
        entropies = []
        rewards: List[float] = []

        sim_step = 0
        total_wait_sum = 0.0
        decisions = 0

        while sim_step < steps:
            vehicles_per_lane = get_vehicle_numbers(lanes)
            state = build_phase_state(vehicles_per_lane, phases)

            out = agent.act(
                state,
                total_time=cycle_time,
                min_green=min_green,
                deterministic=(not train),
            )

            # Log the chosen durations so you can verify the policy is not stuck at 30/30/30/30.
            # In training: print every decision. In test: print only first few decisions.
            sumo_phase_indices = [p["index"] for p in phases]
            durations_list = [int(x) for x in out.durations.tolist()]
            state_list = [float(x) for x in state.tolist()]
            mode = "train" if train else "test"
            print(
                f"[{mode}] Episode {episode+1} | decision {decisions+1} | "
                f"state(veh/phase)={state_list} | cycle={cycle_time}s | durations={durations_list} | sumo_phases={sumo_phase_indices}"
            )

            cycle_wait_sum = 0.0
            cycle_steps = 0

            for phase_local_index, duration in enumerate(out.durations.tolist()):
                phase_index = phases[phase_local_index]["index"]
                set_phase_by_index(junction, phase_index, int(duration))

                for _ in range(int(duration)):
                    if sim_step >= steps:
                        break
                    traci.simulationStep()
                    waiting = float(get_waiting_time(lanes))
                    cycle_wait_sum += waiting
                    total_wait_sum += waiting
                    sim_step += 1
                    cycle_steps += 1

                if sim_step >= steps:
                    break

            # Reward: minimize halting vehicles during the executed part of the cycle.
            reward = -cycle_wait_sum / max(1, cycle_steps)

            if train:
                if out.log_prob is not None:
                    log_probs.append(out.log_prob)
                values.append(out.value.squeeze(-1))
                if out.entropy is not None:
                    entropies.append(out.entropy)
                rewards.append(float(reward))

            decisions += 1

        if train and rewards:
            returns = _compute_returns(rewards, agent.gamma)
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=agent.device)
            # Normalize returns to make optimization less sensitive to episode length/scale.
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std(unbiased=False) + 1e-8)
            log_probs_t = torch.stack(log_probs).to(agent.device)
            values_t = torch.stack(values).to(agent.device)
            entropies_t = torch.stack(entropies).to(agent.device) if entropies else None

            loss = agent.update(
                log_probs=log_probs_t,
                values=values_t,
                returns=returns_t,
                entropies=entropies_t,
            )
            print(
                f"Episode {episode+1}/{episodes} - decisions={decisions} - total_wait_sum={total_wait_sum:.1f} - loss={loss:.4f}"
            )
        else:
            print(
                f"Episode {episode+1}/{episodes} - decisions={decisions} - total_wait_sum={total_wait_sum:.1f}"
            )

        traci.close()

    if train:
        agent.save(model_path)
        print(f"Model saved to {model_path}")
