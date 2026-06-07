import argparse

from schemas import ExperimentConfig
from training.trainer import GraphTrainer
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a graph classifier")
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
        default=None,
        help="Override model architecture",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "sum", "attention"],
        default=None,
        help="Override graph pooling method",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.architecture is not None:
        config.model.architecture = args.architecture
    if args.pooling is not None:
        config.model.pooling = args.pooling
    trainer = GraphTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
