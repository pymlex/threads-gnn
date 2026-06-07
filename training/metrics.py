import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from torch import Tensor


def softmax_probabilities(logits: Tensor) -> np.ndarray:
    """Convert logits to positive-class probabilities in float32."""
    probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
    return probabilities.detach().cpu().numpy()


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics for binary graph labels."""
    return {
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, object]:
    """Save raw and normalised confusion matrices and figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = confusion_matrix(labels, predictions, labels=[0, 1])
    normalised = counts.astype(np.float64) / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    counts_path = output_dir / f"{prefix}_confusion_counts.csv"
    normalised_path = output_dir / f"{prefix}_confusion_normalised.csv"
    pd.DataFrame(counts, columns=[0, 1], index=[0, 1]).to_csv(counts_path)
    pd.DataFrame(normalised, columns=[0, 1], index=[0, 1]).to_csv(normalised_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Confusion counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(normalised, annot=True, fmt=".2f", cmap="Blues", ax=axes[1], cbar=False)
    axes[1].set_title("Confusion normalised")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    figure_path = output_dir / f"{prefix}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    return {
        "counts": counts.tolist(),
        "normalised": normalised.tolist(),
        "counts_path": str(counts_path),
        "normalised_path": str(normalised_path),
        "figure_path": str(figure_path),
    }


def save_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: str | Path,
) -> str:
    """Save sklearn classification report to text."""
    output_path = Path(output_path)
    report = classification_report(labels, predictions, digits=4)
    output_path.write_text(report, encoding="utf-8")
    return report


def save_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    graph_ids: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save test-set predictions to CSV."""
    frame = pd.DataFrame({
        "graph_id": graph_ids,
        "label": labels,
        "prediction": predictions,
        "probability_positive": probabilities,
    })
    frame.to_csv(output_path, index=False)


def append_epoch_metrics(
    metrics_rows: list[dict[str, object]],
    output_path: str | Path,
) -> None:
    """Append per-epoch metrics to CSV."""
    frame = pd.DataFrame(metrics_rows)
    output_path = Path(output_path)
    if output_path.exists():
        frame.to_csv(output_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(output_path, index=False)


def save_json_metrics(payload: dict[str, object], output_path: str | Path) -> None:
    """Save metrics dictionary to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
