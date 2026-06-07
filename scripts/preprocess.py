import argparse
from pathlib import Path

from data.download import download_dataset
from data.preprocess import preprocess_dataset
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess Reddit Threads")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    download_dataset(config.data.raw_dir)
    preprocess_dataset(config)


if __name__ == "__main__":
    main()
