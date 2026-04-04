import optparse
from plain import run_plain
from dqn.train import run_dqn
from heuristic import run_heuristic
from policy.train import run_policy


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option(
        "-m",
        dest="model_name",
        type="string",
        default="model",
        help="Name of the model (default: model)",
    )
    opt_parser.add_option(
        "--train",
        action="store_true",
        default=False,
        help="Train a new DQN model",
    )
    opt_parser.add_option(
        "--test",
        action="store_true",
        default=False,
        help="Run a trained DQN model (opens sumo-gui)",
    )

    opt_parser.add_option(
        "--policy-train",
        action="store_true",
        default=False,
        help="Train a policy-gradient model to allocate a fixed cycle time among 4 green phases",
    )
    opt_parser.add_option(
        "--policy-test",
        action="store_true",
        default=False,
        help="Run a trained policy-gradient model (opens sumo-gui)",
    )
    opt_parser.add_option(
        "--heuristic",
        action="store_true",
        default=False,
        help="Run heuristic mode: green for the lane with the most cars",
    )
    opt_parser.add_option(
        "--plain",
        action="store_true",
        default=False,
        help="Run SUMO config without DQN or heuristic control",
    )
    opt_parser.add_option(
        "-e",
        dest="epochs",
        type="int",
        default=50,
        help="Number of epochs (default: 50)",
    )
    opt_parser.add_option(
        "-s",
        dest="steps",
        type="int",
        default=500,
        help="Number of steps per epoch (default: 500)",
    )

    opt_parser.add_option(
        "--cycle-time",
        dest="cycle_time",
        type="int",
        default=120,
        help="Cycle time (seconds) to distribute among 4 phases in policy mode (default: 120)",
    )
    opt_parser.add_option(
        "--min-green",
        dest="min_green",
        type="int",
        default=5,
        help="Minimum green duration per phase in policy mode (default: 5)",
    )
    options, args = opt_parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()

    if options.plain:
        run_plain(steps=options.steps)
    elif options.heuristic:
        run_heuristic(steps=options.steps)
    elif options.train:
        run_dqn(
            episodes=options.epochs,
            steps=options.steps,
            train=True,
            model_name=options.model_name,
            gui=False,
        )
    elif options.test:
        run_dqn(
            train=False,
            model_name=options.model_name,
            episodes=1,
            steps=options.steps,
            gui=True,
        )
    elif options.policy_train:
        run_policy(
            episodes=options.epochs,
            steps=options.steps,
            train=True,
            model_name=options.model_name,
            gui=False,
            cycle_time=options.cycle_time,
            min_green=options.min_green,
        )
    elif options.policy_test:
        run_policy(
            episodes=1,
            steps=options.steps,
            train=False,
            model_name=options.model_name,
            gui=True,
            cycle_time=options.cycle_time,
            min_green=options.min_green,
        )
    else:
        print(
            "Please specify a mode: --plain, --train, --test, --heuristic, --policy-train, or --policy-test"
        )
        print("Examples:")
        print("  python main.py --plain -s 500")
        print("  python main.py --heuristic -s 500")
        print("  python main.py --train -e 50 -s 500 -m my_model")
        print("  python main.py --test -m my_model")
        print("  python main.py --policy-train -e 50 -s 2000 -m my_policy --cycle-time 120")
        print("  python main.py --policy-test -m my_policy --cycle-time 120")

