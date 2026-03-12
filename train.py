import traci
import os
import torch
from sumolib import checkBinary
from sumo_utils import get_vehicle_numbers, get_waiting_time, get_green_phases, set_phase_by_index
from agent import DQNAgent
import numpy as np

def build_state(vehicles_per_lane, lanes):
    return np.array([vehicles_per_lane.get(lane, 0) for lane in lanes], dtype=np.float32)

def run_dqn(
    episodes=50,
    steps=500,
    min_duration=15,
    batch_size=32,
    train=True,
    model_name="model",
    gui=False,
):

    # Start SUMO once to get lanes and phases
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg"])
    junction = traci.trafficlight.getIDList()[0]
    phases = get_green_phases(junction)
    lanes = sorted(set(lane for phase in phases for lane in phase["lanes"]))
    state_size = len(lanes)
    action_size = len(phases)
    agent = DQNAgent(state_size, action_size)
    model_path = os.path.join("models", f"{model_name}.pth")

    if not train:
        if not os.path.exists(model_path):
            traci.close()
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train first with --train -m {model_name}."
            )
        agent.load(model_path, map_location=torch.device("cpu"))
        agent.epsilon = 0.0

    traci.close()

    sim_binary = "sumo-gui" if gui else "sumo"

    for episode in range(episodes):
        traci.start([checkBinary(sim_binary), "-c", "configuration.sumocfg"])
        state = np.zeros(state_size, dtype=np.float32)
        phase_timer = 0
        total_wait = 0
        action = 0

        for step in range(steps):

            if phase_timer <= 0:
                action = agent.act(state.reshape(1, -1))
                phase_index = phases[action]["index"]
                set_phase_by_index(junction, phase_index, min_duration)
                phase_timer = min_duration

            traci.simulationStep()
            vehicles_per_lane = get_vehicle_numbers(lanes)
            next_state = build_state(vehicles_per_lane, lanes)
            waiting = get_waiting_time(lanes)
            reward = -waiting
            done = step == steps - 1

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_wait += waiting
            phase_timer -= 1

            if train:
                agent.replay(batch_size)

        print(f"Episode {episode+1}/{episodes} - Total waiting: {total_wait}")
        traci.close()

    if train:
        agent.save(model_path)
        print(f"Model saved to {model_path}")