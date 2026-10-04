"""Brief original-mode checks using existing checkpoints; never train."""
import json
import os

from sync.common import ROOT, SUMO_HOME, traci
from tests.validate_sync import REFERENCE, protected


def main():
    os.environ.setdefault("SUMO_HOME", str(SUMO_HOME))
    os.environ["PATH"] = str(SUMO_HOME / "bin") + os.pathsep + os.environ["PATH"]
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    protected(reference)
    from policy.train import run_policy
    from dqn.train import run_dqn
    from plain import run_plain
    from heuristic import run_heuristic
    run_policy(episodes=1, steps=120, train=False,
               model_name="model_future_v39_yellow_test36_3", gui=False,
               min_green=5, max_green=45)
    run_dqn(episodes=1, steps=120, train=False, model_name="policy", gui=False)
    # The original GUI modes expect a manual Play click. Automate only that
    # launch option in this smoke fixture; their code and control stay intact.
    original_start = traci.start
    def auto_play(command, *args, **kwargs):
        return original_start(command + ["--start", "--quit-on-end"], *args, **kwargs)
    traci.start = auto_play
    try:
        lane_metrics = run_plain(steps=120)
        assert set(lane_metrics) == {"waiting_time", "effective_flow", "avg_queue_length"}
        run_heuristic(steps=120)
    finally:
        traci.start = original_start
    protected(reference)
    output = ROOT / "results/original_smoke.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"steps_requested": 120, "PPO": "passed", "DQN": "passed",
        "plain": "passed", "heuristic": "passed", "training": False,
        "protected_hashes_unchanged": True, "plain_lane_metrics": lane_metrics}, indent=2), encoding="utf-8")
    print("PASS original PPO, DQN, plain and heuristic; protected files unchanged")


if __name__ == "__main__":
    main()
