import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from epoch metrics")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing run outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in experiments",
    )
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    architectures = ["gin", "pna", "gat"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for architecture, axis in zip(architectures, axes):
        metrics_path = runs_dir / f"{architecture}_seed{args.seed}" / "epoch_metrics.csv"
        frame = pd.read_csv(metrics_path)
        axis.plot(frame["epoch"], frame["train_mcc"], label="train MCC")
        axis.plot(frame["epoch"], frame["val_mcc"], label="val MCC")
        axis.plot(frame["epoch"], frame["test_mcc"], label="test MCC")
        axis.set_title(architecture.upper())
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MCC")
        axis.grid(alpha=0.5)
        axis.legend()

    output_path = runs_dir / "training_curves.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
