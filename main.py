import optparse
from plain import run_plain
from train import run_dqn
from heuristic import run_heuristic


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
    else:
        print("Please specify a mode: --plain, --train, --test, or --heuristic")
        print("Examples:")
        print("  python main.py --plain -s 500")
        print("  python main.py --heuristic -s 500")
        print("  python main.py --train -e 50 -s 500 -m my_model")
        print("  python main.py --test -m my_model")

