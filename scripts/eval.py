import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.dataset import RedditThreadsDataset, collate_graphs
from features.engineering import feature_dim
from models import build_model
from training.metrics import (
    compute_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_json_metrics,
    save_predictions,
    softmax_probabilities,
)
from utils.config import load_config


ARCHITECTURES = ["gin", "pna", "gat"]


def evaluate_checkpoint(
    config,
    checkpoint_path: Path,
    architecture: str,
    split: str,
    seed: int,
) -> dict[str, float]:
    """Evaluate one checkpoint on a dataset split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.model.architecture = architecture

    dataset = RedditThreadsDataset(config.data.processed_dir, split)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_graphs,
    )

    train_loader = DataLoader(
        RedditThreadsDataset(config.data.processed_dir, "train"),
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
    predictions_all = []
    probabilities_all = []
    graph_ids_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {architecture} {split}", leave=False):
            batch = batch.to(device)
            logits = model(batch)
            probabilities = softmax_probabilities(logits)
            predictions = logits.argmax(dim=-1).cpu().numpy()
            labels = batch.y.view(-1).cpu().numpy()
            graph_ids = [int(item.graph_id) for item in batch.to_data_list()]
            labels_all.append(labels)
            predictions_all.append(predictions)
            probabilities_all.append(probabilities)
            graph_ids_all.extend(graph_ids)

    labels_np = np.concatenate(labels_all)
    predictions_np = np.concatenate(predictions_all)
    probabilities_np = np.concatenate(probabilities_all)
    graph_ids_np = np.array(graph_ids_all)

    metrics = compute_metrics(labels_np, predictions_np, probabilities_np)
    run_dir = Path(config.output.runs_dir) / f"{architecture}_seed{seed}" / f"eval_{split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(labels_np, predictions_np, run_dir, split)
    save_classification_report(
        labels_np,
        predictions_np,
        run_dir / f"{split}_classification_report.txt",
    )
    save_predictions(
        labels_np,
        predictions_np,
        probabilities_np,
        graph_ids_np,
        run_dir / f"{split}_predictions.csv",
    )
    save_json_metrics(metrics, run_dir / f"{split}_metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["all", "gin", "pna", "gat"],
        default="all",
        help="Architecture to evaluate or all for every trained model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for a single architecture evaluation",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in checkpoint filenames",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    architectures = ARCHITECTURES if args.architecture == "all" else [args.architecture]
    results = {}

    for architecture in architectures:
        if args.checkpoint is not None and args.architecture != "all":
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = (
                Path(config.output.checkpoints_dir)
                / f"{architecture}_seed{args.seed}_best.pt"
            )
        metrics = evaluate_checkpoint(
            config,
            checkpoint_path,
            architecture,
            args.split,
            args.seed,
        )
        results[architecture] = metrics
        tqdm.write(f"{architecture} {args.split} MCC {metrics['mcc']:.4f}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
