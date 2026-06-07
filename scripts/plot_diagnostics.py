import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.dataset import RedditThreadsDataset, ShardStore, collate_graphs
from features.engineering import feature_dim
from models import build_model
from training.metrics import softmax_probabilities
from utils.config import load_config


ARCHITECTURES = ["gin", "pna", "gat"]
GITHUB_RAW = "https://raw.githubusercontent.com/pymlex/threads-gnn/main"


def collect_predictions(
    config,
    architecture: str,
    checkpoint_path: Path,
    split: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run a saved checkpoint and collect labels, logits, and probabilities."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.model.architecture = architecture

    processed_dir = Path(config.data.processed_dir)
    splits_path = processed_dir / "splits.json"
    shard_store = ShardStore(processed_dir)
    dataset = RedditThreadsDataset(shard_store, split, splits_path)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_graphs,
    )

    train_loader = DataLoader(
        RedditThreadsDataset(shard_store, "train", splits_path),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_graphs,
    ) if architecture == "pna" else None

    input_dim = feature_dim(config.features)
    model = build_model(
        architecture,
        input_dim,
        config.model,
        train_loader,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels_all = []
    logits_all = []
    probabilities_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predict {architecture} {split}", leave=False):
            batch = batch.to(device)
            logits = model(batch).float()
            probabilities = softmax_probabilities(logits)
            labels = batch.y.view(-1).cpu().numpy()
            labels_all.append(labels)
            logits_all.append(logits.cpu().numpy())
            probabilities_all.append(probabilities)

    return {
        "labels": np.concatenate(labels_all),
        "logits": np.concatenate(logits_all),
        "probabilities": np.concatenate(probabilities_all),
    }


def plot_logit_histogram(
    labels: np.ndarray,
    logits: np.ndarray,
    architecture: str,
    output_path: Path,
) -> None:
    """Plot class-1 logit distribution split by ground-truth label."""
    logit_positive = logits[:, 1]
    label_zero = logit_positive[labels == 0]
    label_one = logit_positive[labels == 1]

    fig, axis = plt.subplots(figsize=(5, 4))
    axis.hist(
        label_zero,
        bins=50,
        alpha=0.55,
        density=True,
        label="true class 0",
    )
    axis.hist(
        label_one,
        bins=50,
        alpha=0.55,
        density=True,
        label="true class 1",
    )
    axis.set_title(f"{architecture.upper()} test logit class 1")
    axis.set_xlabel("logit")
    axis.set_ylabel("density")
    axis.grid(alpha=0.5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    architecture: str,
    output_path: Path,
) -> float:
    """Plot ROC curve for one architecture and return ROC-AUC."""
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{architecture.upper()} AUC = {roc_auc:.4f}",
    )
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.45")
    axis.set_title(f"{architecture.upper()} test ROC")
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.grid(alpha=0.5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return float(roc_auc)


def plot_combined_logit_histograms(
    predictions: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot logit histograms for all architectures in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for architecture, axis in zip(ARCHITECTURES, axes):
        labels = predictions[architecture]["labels"]
        logit_positive = predictions[architecture]["logits"][:, 1]
        axis.hist(
            logit_positive[labels == 0],
            bins=50,
            alpha=0.55,
            density=True,
            label="true class 0",
        )
        axis.hist(
            logit_positive[labels == 1],
            bins=50,
            alpha=0.55,
            density=True,
            label="true class 1",
        )
        axis.set_title(architecture.upper())
        axis.set_xlabel("logit class 1")
        axis.set_ylabel("density")
        axis.grid(alpha=0.5)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_combined_roc_curves(
    predictions: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot ROC curves for all architectures in one figure."""
    fig, axis = plt.subplots(figsize=(6, 5))
    for architecture in ARCHITECTURES:
        labels = predictions[architecture]["labels"]
        probabilities = predictions[architecture]["probabilities"]
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{architecture.upper()} AUC = {roc_auc:.4f}",
        )
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.45")
    axis.set_title("Test ROC curves")
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.grid(alpha=0.5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot logit histograms and ROC curves")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    runs_dir = Path(args.runs_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    predictions: dict[str, dict[str, np.ndarray]] = {}

    for architecture in ARCHITECTURES:
        checkpoint_path = (
            checkpoints_dir / f"{architecture}_seed{args.seed}_best.pt"
        )
        run_dir = runs_dir / f"{architecture}_seed{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = collect_predictions(
            config,
            architecture,
            checkpoint_path,
            args.split,
            args.seed,
        )
        predictions[architecture] = payload
        plot_logit_histogram(
            payload["labels"],
            payload["logits"],
            architecture,
            run_dir / f"{args.split}_logit_histogram.png",
        )
        plot_roc_curve(
            payload["labels"],
            payload["probabilities"],
            architecture,
            run_dir / f"{args.split}_roc_curve.png",
        )

    plot_combined_logit_histograms(
        predictions,
        runs_dir / f"{args.split}_logit_histograms.png",
    )
    plot_combined_roc_curves(
        predictions,
        runs_dir / f"{args.split}_roc_curves.png",
    )

    tqdm.write(f"Saved diagnostics to {runs_dir}")
    tqdm.write(f"GitHub raw base for figures: {GITHUB_RAW}/runs/")


if __name__ == "__main__":
    main()
