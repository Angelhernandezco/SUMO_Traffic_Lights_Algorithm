from sumolib import checkBinary
import traci

from sumo_utils import get_lane_metrics


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

    # Only evaluate metrics on lanes controlled by traffic lights.
    traffic_lights = traci.trafficlight.getIDList()
    traffic_light_lanes = set()
    for tl_id in traffic_lights:
        traffic_light_lanes.update(traci.trafficlight.getControlledLanes(tl_id))
    traffic_light_lanes = list(traffic_light_lanes)

    delta_t = traci.simulation.getDeltaT()  # seconds per step (usually 1.0)

    step = 0
    total_time = 0        # stopped vehicles summed per step (original "waiting time")
    effective_flow = 0.0  # vehicle-seconds in motion
    queue_sum = 0.0       # sum of per-step average queue length

    while step <= steps and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        m = get_lane_metrics(traffic_light_lanes)

        total_time += m["halting"]
        effective_flow += m["moving"] * delta_t
        queue_sum += m["avg_queue"]
        step += 1

    traci.close()

    avg_queue_length = queue_sum / step if step else 0.0

    print("Plain total waiting time:", total_time)
    print("Plain effective flow (veh·s in motion):", effective_flow)
    print("Plain average queue length (veh/lane):", avg_queue_length)

    return {
        "waiting_time": total_time,
        "effective_flow": effective_flow,
        "avg_queue_length": avg_queue_length,
    }