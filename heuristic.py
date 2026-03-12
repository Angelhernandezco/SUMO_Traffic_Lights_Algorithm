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
    phase_lanes_map = {
        junction: {phase["index"]: phase["lanes"] for phase in phase_map.get(junction, [])}
        for junction in all_junctions
    }

    traffic_light_duration = 30
    min_duration = 10
    max_seconds_without_green = 180

    traffic_lights_time = {junction: 0 for junction in all_junctions}
    current_green_phase = {junction: None for junction in all_junctions}
    phase_red_time = {
        junction: {phase["index"]: 0 for phase in phase_map.get(junction, [])}
        for junction in all_junctions
    }

    step = 0    # Simulation step counter
    total_time = 0  # Accumulator for total vehicle waiting time

    while step <= steps:
        for junction in all_junctions:
            should_change_phase = traffic_lights_time[junction] <= 0

            # Change early if the current green phase has no vehicles left.
            if not should_change_phase and current_green_phase[junction] is not None:
                active_phase_lanes = phase_lanes_map[junction].get(current_green_phase[junction], [])
                if active_phase_lanes:
                    active_phase_vehicles = get_vehicle_numbers(list(set(active_phase_lanes)))
                    vehicles_in_active_phase = sum(
                        active_phase_vehicles.get(lane_id, 0) for lane_id in active_phase_lanes
                    )
                    elapsed_green_time = traffic_light_duration - traffic_lights_time[junction]
                    if vehicles_in_active_phase == 0 and elapsed_green_time >= min_duration:
                        should_change_phase = True

            if should_change_phase:
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

                    # Force green phase for red phases lasting beyond the configured threshold
                    forced_phase = None
                    overdue_candidates = []
                    for i, phase in enumerate(junction_phases):
                        phase_index = phase["index"]
                        waiting_vehicles = phase_totals[i]
                        red_time = phase_red_time[junction].get(phase_index, 0)
                        if (
                            waiting_vehicles > 0
                            and red_time >= max_seconds_without_green
                        ):
                            overdue_candidates.append((red_time, waiting_vehicles, phase_index))

                    if overdue_candidates:
                        # Prioritize the most lasting red phase. [2] is for obtaining the phase index from the tuple.
                        forced_phase = max(overdue_candidates)[2]

                    # Choose forced phase (if any) or fallback to highest vehicle count
                    if forced_phase is not None:
                        best_phase = forced_phase
                    else:
                        best_phase_idx = int(np.argmax(phase_totals))
                        best_phase = junction_phases[best_phase_idx]["index"]

                    # Activate the chosen phase for a fixed duration
                    set_phase_by_index(junction, best_phase, traffic_light_duration)
                    traffic_lights_time[junction] = traffic_light_duration
                    current_green_phase[junction] = best_phase
                else:
                    # Fallback if no phases were detected
                    traffic_lights_time[junction] = 1

        # Advance the simulation by one step (1 second)
        traci.simulationStep()

        for junction in all_junctions:
            # Get all lanes controlled by this traffic light
            controlled_lanes = traci.trafficlight.getControlledLanes(junction)
            # Compute the total waiting time in those lanes
            waiting_time = get_waiting_time(controlled_lanes)
            total_time += waiting_time

            # Track how long each phase has remained without green.
            for phase in phase_map.get(junction, []):
                phase_index = phase["index"]
                if current_green_phase[junction] == phase_index:
                    phase_red_time[junction][phase_index] = 0
                else:
                    phase_red_time[junction][phase_index] += 1

            # Consume one second of the active phase timer
            traffic_lights_time[junction] -= 1

        step += 1

    print("Heuristic total waiting time:", total_time)
    traci.close()
