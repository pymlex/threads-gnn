import argparse
import json
from pathlib import Path

import pandas as pd

from schemas import ExperimentConfig
from training.trainer import GraphTrainer
from utils.config import load_config


POOLING_METHODS = ["mean", "sum", "attention"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pooling methods on validation MCC")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["gin", "pna", "gat"],
        default="gin",
        help="Architecture used for pooling comparison",
    )
    args = parser.parse_args()
    base_config = load_config(args.config)
    runs_dir = Path(base_config.output.runs_dir) / "pooling_comparison"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pooling in POOLING_METHODS:
        config = ExperimentConfig.model_validate(base_config.model_dump())
        config.model.architecture = args.architecture
        config.model.pooling = pooling
        config.output.runs_dir = str(runs_dir)
        trainer = GraphTrainer(config)
        summary = trainer.train()
        rows.append({
            "pooling": pooling,
            "best_val_mcc": summary["best_val_mcc"],
            "val_mcc": summary["val_metrics"]["mcc"],
            "test_mcc": summary["test_metrics"]["mcc"],
        })

    frame = pd.DataFrame(rows).sort_values("best_val_mcc", ascending=False)
    frame.to_csv(runs_dir / "pooling_comparison.csv", index=False)

    selection = {
        "selected_pooling": frame.iloc[0]["pooling"],
        "selection_metric": "validation_mcc",
        "comparison_table": frame.to_dict(orient="records"),
    }
    with (runs_dir / "selected_pooling.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)

    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
