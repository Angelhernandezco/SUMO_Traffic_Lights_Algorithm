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

    step = 0
    total_time = 0
    while step <= steps and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        all_lanes = traci.lane.getIDList()
        total_time += get_waiting_time(all_lanes)
        step += 1

    traci.close()
    print("Plain total waiting time:", total_time)
    return total_time
