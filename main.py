import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Threads-GNN experiment entry point")
    parser.add_argument(
        "command",
        choices=["preprocess", "train", "eval", "compare", "run_all", "push_hf"],
        help="Pipeline command to execute",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args, remainder = parser.parse_known_args()
    python = sys.executable

    if args.command == "preprocess":
        command = [python, "scripts/preprocess.py", "--config", args.config]
    elif args.command == "train":
        command = [python, "scripts/train.py", "--config", args.config]
        if args.architecture is not None:
            command.extend(["--architecture", args.architecture])
    elif args.command == "eval":
        command = [python, "scripts/eval.py", "--config", args.config]
        if args.checkpoint is not None:
            command.extend(["--checkpoint", args.checkpoint])
    elif args.command == "compare":
        command = [python, "scripts/compare.py"]
    elif args.command == "run_all":
        command = [python, "scripts/run_all.py", "--config", args.config]
    elif args.command == "push_hf":
        command = [python, "scripts/push_hf.py"]
    else:
        raise ValueError(f"Unknown command: {args.command}")

    command.extend(remainder)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
