import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.dataset import RedditThreadsDataset, collate_graphs
from features.engineering import feature_dim
from models import build_model
from schemas import ExperimentConfig
from training.metrics import (
    append_epoch_metrics,
    compute_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_json_metrics,
    save_predictions,
    softmax_probabilities,
)
from utils.config import save_config
from utils.seed import set_seed


class GraphTrainer:
    """Training and evaluation loop for graph classifiers."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        set_seed(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processed_dir = Path(config.data.processed_dir)
        self.run_dir = (
            Path(config.output.runs_dir)
            / f"{config.model.architecture}_seed{config.seed}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.run_dir / "config.json")

        self.train_loader, self.val_loader, self.test_loader = self._build_loaders()
        input_dim = feature_dim(config.features)
        self.model = build_model(
            config.model.architecture,
            input_dim,
            config.model,
            self.train_loader,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_val_mcc = -np.inf
        self.patience_counter = 0
        self.epoch_metrics_path = self.run_dir / "epoch_metrics.csv"
        self.history: list[dict[str, object]] = []

    def _build_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders."""
        batch_size = self.config.training.batch_size
        num_workers = self.config.training.num_workers
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_graphs,
            "pin_memory": self.device.type == "cuda",
        }
        train_dataset = RedditThreadsDataset(self.processed_dir, "train")
        val_dataset = RedditThreadsDataset(self.processed_dir, "val")
        test_dataset = RedditThreadsDataset(self.processed_dir, "test")
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        return train_loader, val_loader, test_loader

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning-rate scheduler."""
        if self.config.training.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_epochs,
        )

    def _forward_batch(self, batch) -> torch.Tensor:
        batch = batch.to(self.device)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            return self.model(batch)

    def _evaluate_loader(self, loader: DataLoader, desc: str) -> dict[str, object]:
        """Evaluate model on a dataloader."""
        self.model.eval()
        labels_all = []
        predictions_all = []
        probabilities_all = []
        graph_ids_all = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                logits = self._forward_batch(batch)
                probabilities = softmax_probabilities(logits)
                predictions = logits.argmax(dim=-1).cpu().numpy()
                labels = batch.y.view(-1).cpu().numpy()
                graph_ids = np.array([
                    int(item.graph_id) for item in batch.to_data_list()
                ])

                labels_all.append(labels)
                predictions_all.append(predictions)
                probabilities_all.append(probabilities)
                graph_ids_all.append(graph_ids)

        labels_np = np.concatenate(labels_all)
        predictions_np = np.concatenate(predictions_all)
        probabilities_np = np.concatenate(probabilities_all)
        graph_ids_np = np.concatenate(graph_ids_all)
        metrics = compute_metrics(labels_np, predictions_np, probabilities_np)
        return {
            "metrics": metrics,
            "labels": labels_np,
            "predictions": predictions_np,
            "probabilities": probabilities_np,
            "graph_ids": graph_ids_np,
        }

    def _train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        losses = []
        labels_all = []
        predictions_all = []
        probabilities_all = []

        for batch in tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        ):
            self.optimizer.zero_grad(set_to_none=True)
            batch = batch.to(self.device)
            labels = batch.y.view(-1)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(batch)
                loss = F.cross_entropy(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(float(loss.item()))
            probabilities = softmax_probabilities(logits)
            predictions = logits.argmax(dim=-1).cpu().numpy()
            labels_all.append(labels.cpu().numpy())
            predictions_all.append(predictions)
            probabilities_all.append(probabilities)

        labels_np = np.concatenate(labels_all)
        predictions_np = np.concatenate(predictions_all)
        probabilities_np = np.concatenate(probabilities_all)
        metrics = compute_metrics(labels_np, predictions_np, probabilities_np)
        metrics["loss"] = float(np.mean(losses))
        return metrics

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        checkpoint_dir = Path(self.config.output.checkpoints_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_mcc": self.best_val_mcc,
            "config": self.config.model_dump(),
        }
        latest_path = (
            checkpoint_dir
            / f"{self.config.model.architecture}_seed{self.config.seed}_latest.pt"
        )
        torch.save(payload, latest_path)
        if is_best:
            best_path = (
                checkpoint_dir
                / f"{self.config.model.architecture}_seed{self.config.seed}_best.pt"
            )
            torch.save(payload, best_path)

    def train(self) -> dict[str, object]:
        """Run full training with early stopping on validation MCC."""
        for epoch in range(1, self.config.training.num_epochs + 1):
            train_metrics = self._train_epoch()
            val_result = self._evaluate_loader(self.val_loader, "Validation")
            test_result = self._evaluate_loader(self.test_loader, "Test")
            val_metrics = val_result["metrics"]
            test_metrics = test_result["metrics"]

            if self.config.training.scheduler == "plateau":
                self.scheduler.step(val_metrics["mcc"])
            else:
                self.scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mcc": train_metrics["mcc"],
                "train_accuracy": train_metrics["accuracy"],
                "val_mcc": val_metrics["mcc"],
                "val_accuracy": val_metrics["accuracy"],
                "test_mcc": test_metrics["mcc"],
                "test_accuracy": test_metrics["accuracy"],
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(row)
            append_epoch_metrics([row], self.epoch_metrics_path)

            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"train MCC {train_metrics['mcc']:.4f} | "
                f"val MCC {val_metrics['mcc']:.4f} | "
                f"test MCC {test_metrics['mcc']:.4f}"
            )

            is_best = val_metrics["mcc"] > self.best_val_mcc
            if is_best:
                self.best_val_mcc = val_metrics["mcc"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self._save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.config.training.early_stopping_patience:
                tqdm.write(
                    f"Early stopping at epoch {epoch} "
                    f"with best val MCC {self.best_val_mcc:.4f}"
                )
                break

        best_checkpoint = (
            Path(self.config.output.checkpoints_dir)
            / f"{self.config.model.architecture}_seed{self.config.seed}_best.pt"
        )
        checkpoint = torch.load(best_checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        val_result = self._evaluate_loader(self.val_loader, "Validation")
        test_result = self._evaluate_loader(self.test_loader, "Test")

        confusion_info = save_confusion_matrix(
            test_result["labels"],
            test_result["predictions"],
            self.run_dir,
            "test",
        )
        report = save_classification_report(
            test_result["labels"],
            test_result["predictions"],
            self.run_dir / "test_classification_report.txt",
        )
        save_predictions(
            test_result["labels"],
            test_result["predictions"],
            test_result["probabilities"],
            test_result["graph_ids"],
            self.run_dir / "test_predictions.csv",
        )

        summary = {
            "architecture": self.config.model.architecture,
            "seed": self.config.seed,
            "best_val_mcc": self.best_val_mcc,
            "val_metrics": val_result["metrics"],
            "test_metrics": test_result["metrics"],
            "confusion_matrix": confusion_info,
            "classification_report": report,
        }
        save_json_metrics(summary, self.run_dir / "final_metrics.json")
        return summary
