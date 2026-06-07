import argparse
import json
from pathlib import Path

import pandas as pd


ARCHITECTURES = ["gin", "pna", "gat"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare architectures by validation MCC")
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

    rows = []
    for architecture in ARCHITECTURES:
        metrics_path = runs_dir / f"{architecture}_seed{args.seed}" / "final_metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append({
            "architecture": architecture,
            "best_val_mcc": payload["best_val_mcc"],
            "val_mcc": payload["val_metrics"]["mcc"],
            "val_f1": payload["val_metrics"]["f1"],
            "val_roc_auc": payload["val_metrics"]["roc_auc"],
            "test_mcc": payload["test_metrics"]["mcc"],
            "test_f1": payload["test_metrics"]["f1"],
            "test_roc_auc": payload["test_metrics"]["roc_auc"],
        })

    frame = pd.DataFrame(rows).sort_values("best_val_mcc", ascending=False)
    output_path = runs_dir / "architecture_comparison.csv"
    frame.to_csv(output_path, index=False)

    best_row = frame.iloc[0]
    selection = {
        "selected_architecture": best_row["architecture"],
        "selection_metric": "validation_mcc",
        "best_val_mcc": float(best_row["best_val_mcc"]),
        "comparison_table": frame.to_dict(orient="records"),
    }
    selection_path = runs_dir / "selected_model.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)

    print(frame.to_string(index=False))
    print(f"\nSelected architecture: {best_row['architecture']}")


if __name__ == "__main__":
    main()
