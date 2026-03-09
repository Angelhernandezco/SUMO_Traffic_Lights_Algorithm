from sumolib import checkBinary
import traci

from sumo_utils import get_waiting_time


def run_plain(steps=500):
    """Run SUMO with the configured network/routes and no custom algorithm."""
    traci.start(
        [
            checkBinary("sumo-gui"),
            "-c",
            "configuration.sumocfg",
            "--tripinfo-output",
            "maps/tripinfo.xml",
        ]
    )

    # Only evaluate waiting time on lanes controlled by traffic lights.
    traffic_lights = traci.trafficlight.getIDList()
    traffic_light_lanes = set()
    for tl_id in traffic_lights:
        traffic_light_lanes.update(traci.trafficlight.getControlledLanes(tl_id))
    traffic_light_lanes = list(traffic_light_lanes)

    step = 0
    total_time = 0
    while step <= steps and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        total_time += get_waiting_time(traffic_light_lanes)
        step += 1

    traci.close()
    print("Plain total waiting time:", total_time)
    return total_time
