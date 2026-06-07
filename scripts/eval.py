import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RedditThreadsDataset(config.data.processed_dir, args.split)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_graphs,
    )

    input_dim = feature_dim(config.features)
    train_loader = None
    if config.model.architecture == "pna" and args.split == "train":
        train_loader = loader
    elif config.model.architecture == "pna":
        train_loader = DataLoader(
            RedditThreadsDataset(config.data.processed_dir, "train"),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=collate_graphs,
        )

    model = build_model(
        config.model.architecture,
        input_dim,
        config.model,
        train_loader,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels_all = []
    predictions_all = []
    probabilities_all = []
    graph_ids_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probabilities = softmax_probabilities(logits)
            predictions = logits.argmax(dim=-1).cpu().numpy()
            labels = batch.y.view(-1).cpu().numpy()
            graph_ids = [
                int(item.graph_id) for item in batch.to_data_list()
            ]
            labels_all.append(labels)
            predictions_all.append(predictions)
            probabilities_all.append(probabilities)
            graph_ids_all.extend(graph_ids)

    labels_np = np.concatenate(labels_all)
    predictions_np = np.concatenate(predictions_all)
    probabilities_np = np.concatenate(probabilities_all)
    graph_ids_np = np.array(graph_ids_all)

    metrics = compute_metrics(labels_np, predictions_np, probabilities_np)
    run_dir = Path(config.output.runs_dir) / f"eval_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(labels_np, predictions_np, run_dir, args.split)
    save_classification_report(
        labels_np,
        predictions_np,
        run_dir / f"{args.split}_classification_report.txt",
    )
    save_predictions(
        labels_np,
        predictions_np,
        probabilities_np,
        graph_ids_np,
        run_dir / f"{args.split}_predictions.csv",
    )
    save_json_metrics(metrics, run_dir / f"{args.split}_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
