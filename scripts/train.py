import argparse

from schemas import ExperimentConfig
from training.trainer import GraphTrainer
from utils.config import load_config


ARCHITECTURES = ["gin", "pna", "gat"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train graph classifiers")
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
        help="Model architecture or all for GIN, PNA, and GAT",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "sum", "attention"],
        default=None,
        help="Override graph pooling method",
    )
    args = parser.parse_args()

    architectures = ARCHITECTURES if args.architecture == "all" else [args.architecture]

    for architecture in architectures:
        config = load_config(args.config)
        config.model.architecture = architecture
        if args.pooling is not None:
            config.model.pooling = args.pooling
        trainer = GraphTrainer(config)
        trainer.train()


if __name__ == "__main__":
    main()
