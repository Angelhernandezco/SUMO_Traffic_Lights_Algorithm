import numpy as np
from sumolib import checkBinary
import traci
from sumo_utils import (
    get_vehicle_numbers,
    get_waiting_time,
    get_green_phases,
    set_phase_by_index,
)


def run_heuristic(steps=500):
    """Run heuristic mode: pick the phase with the highest vehicle count."""

    # Start SUMO GUI using the configuration file
    traci.start(
        [
            checkBinary("sumo-gui"),
            "-c",
            "configuration.sumocfg",
            "--tripinfo-output",
            "maps/tripinfo.xml",
        ]
    )
    # Get the IDs of all traffic lights (junctions) in the simulation
    all_junctions = traci.trafficlight.getIDList()
    phase_map = {junction: get_green_phases(junction) for junction in all_junctions}

    min_duration = 15
    traffic_lights_time = {junction: 0 for junction in all_junctions}

    step = 0    # Simulation step counter
    total_time = 0  # Accumulator for total vehicle waiting time

    while step <= steps:
        for junction in all_junctions:
            if traffic_lights_time[junction] <= 0:
                # Get available green phases for this junction
                junction_phases = phase_map.get(junction, [])

                if junction_phases:
                    # Collect all lanes involved in any possible phase
                    all_lanes_in_phases = []
                    for phase in junction_phases:
                        all_lanes_in_phases.extend(phase["lanes"])

                    # Count vehicles per lane
                    vehicles_per_lane = get_vehicle_numbers(list(set(all_lanes_in_phases)))
                    phase_totals = []

                    # Compute the total number of vehicles for each candidate phase
                    for phase in junction_phases:
                        phase_totals.append(
                            sum(vehicles_per_lane.get(lane_id, 0) for lane_id in phase["lanes"])
                        )

                    # Choose the phase with the highest vehicle count
                    best_phase_idx = int(np.argmax(phase_totals))
                    best_phase = junction_phases[best_phase_idx]["index"]

                    # Activate the chosen phase for a fixed duration
                    set_phase_by_index(junction, best_phase, min_duration)
                    traffic_lights_time[junction] = min_duration
                else:
                    # Fallback if no phases were detected
                    traffic_lights_time[junction] = 1

        # Advance the simulation by one step (usually 1 second)
        traci.simulationStep()

        for junction in all_junctions:
            # Get all lanes controlled by this traffic light
            controlled_lanes = traci.trafficlight.getControlledLanes(junction)
            # Compute the total waiting time in those lanes
            waiting_time = get_waiting_time(controlled_lanes)
            total_time += waiting_time

            # Consume one second of the active phase timer
            traffic_lights_time[junction] -= 1

        step += 1

    print("Heuristic total waiting time:", total_time)
    traci.close()
