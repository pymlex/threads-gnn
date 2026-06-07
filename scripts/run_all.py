import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration",
    )
    args = parser.parse_args()
    python = sys.executable

    subprocess.run(
        [python, "scripts/preprocess.py", "--config", args.config],
        check=True,
    )
    subprocess.run(
        [
            python,
            "scripts/train.py",
            "--config",
            args.config,
            "--architecture",
            "all",
        ],
        check=True,
    )
    subprocess.run([python, "scripts/compare.py"], check=True)
    subprocess.run(
        [python, "scripts/eval.py", "--config", args.config, "--architecture", "all"],
        check=True,
    )
    subprocess.run([python, "scripts/plot_curves.py"], check=True)
    subprocess.run(
        [python, "scripts/plot_diagnostics.py", "--config", args.config],
        check=True,
    )


if __name__ == "__main__":
    main()
