import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload best checkpoint to Hugging Face")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="pymlex/threads-gnn",
        help="Hugging Face repository id",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing experiment outputs",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in experiments",
    )
    args = parser.parse_args()

    selection_path = Path(args.runs_dir) / "selected_model.json"
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)

    architecture = selection["selected_architecture"]
    checkpoint_path = (
        Path(args.checkpoints_dir)
        / f"{architecture}_seed{args.seed}_best.pt"
    )
    run_dir = Path(args.runs_dir) / f"{architecture}_seed{args.seed}"

    api = HfApi()
    create_repo(args.repo_id, exist_ok=True, repo_type="model")

    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="model.pt",
        repo_id=args.repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(run_dir / "config.json"),
        path_in_repo="config.json",
        repo_id=args.repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(run_dir / "final_metrics.json"),
        path_in_repo="final_metrics.json",
        repo_id=args.repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(selection_path),
        path_in_repo="selected_model.json",
        repo_id=args.repo_id,
        repo_type="model",
    )
    model_card = Path("model_card.md")
    card_path = model_card if model_card.exists() else Path("README.md")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )


if __name__ == "__main__":
    main()
