import os
import sys
import traci

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Declare environment variable 'SUMO_HOME'")


def get_green_phases(junction):
    """
    Return the green phases for a traffic light.

    Each phase contains:
    - the SUMO phase index
    - the lanes that receive green light in that phase
    """

    # Get the traffic light logic programs
    logics = traci.trafficlight.getAllProgramLogics(junction)
    if not logics:
        return []

    phases = logics[0].phases
    controlled_links = traci.trafficlight.getControlledLinks(junction)
    green_phases = []

    for phase_index, phase in enumerate(phases):
        state = phase.state

        # Ignore transition/all-red phases; keep only phases that move traffic.
        if "G" not in state and "g" not in state:
            continue
        if "y" in state or "Y" in state:
            continue

        lanes = set()
        for signal_index, signal_state in enumerate(state):
            if signal_state not in ("G", "g"):
                continue
            if signal_index >= len(controlled_links):
                continue

            for link in controlled_links[signal_index]:
                if link and len(link) >= 1:
                    lanes.add(link[0])

        if lanes:
            green_phases.append(
                {
                    "index": phase_index,
                    "lanes": list(lanes),
                }
            )

    return green_phases


def set_phase_by_index(junction, phase_index, phase_time):
    """Switch traffic light to a specific phase index and keep it active for phase_time."""

    traci.trafficlight.setPhase(junction, phase_index)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


def get_vehicle_numbers(lanes):
    """Return a dictionary with the number of vehicles in each lane."""

    vehicle_per_lane = {}
    for lane_id in lanes:
        vehicle_per_lane[lane_id] = traci.lane.getLastStepVehicleNumber(lane_id)
    return vehicle_per_lane


def get_waiting_time(lanes):
    """Compute total vehicles waiting in the given lanes."""

    waiting_time = 0
    for lane_id in lanes:
        waiting_time += traci.lane.getLastStepHaltingNumber(lane_id)
    return waiting_time
