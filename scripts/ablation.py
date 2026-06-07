import argparse
import json
from pathlib import Path

import pandas as pd

from data.download import download_dataset
from data.preprocess import preprocess_dataset
from features.engineering import feature_dim
from schemas import ExperimentConfig, FeatureConfig
from training.trainer import GraphTrainer
from utils.config import load_config


ABLATION_VARIANTS = {
    "full": {},
    "no_laplacian_pe": {"use_laplacian_pe": False},
    "no_rwse": {"use_rwse": False},
    "no_pagerank": {"use_pagerank": False},
    "no_clustering": {"use_clustering": False},
    "no_kcore": {"use_kcore": False},
    "no_degree_bucket": {"use_degree_bucket": False},
    "degree_only": {
        "use_log_degree": False,
        "use_normalised_degree": False,
        "use_degree_bucket": False,
        "use_clustering": False,
        "use_kcore": False,
        "use_pagerank": False,
        "use_laplacian_pe": False,
        "use_rwse": False,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature ablation experiments")
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
        default="gin",
        help="Architecture used for ablation",
    )
    args = parser.parse_args()
    base_config = load_config(args.config)
    ablation_dir = Path(base_config.output.runs_dir) / "feature_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant_name, overrides in ABLATION_VARIANTS.items():
        feature_payload = FeatureConfig().model_dump()
        feature_payload.update(overrides)
        config = ExperimentConfig.model_validate(base_config.model_dump())
        config.features = FeatureConfig.model_validate(feature_payload)
        config.model.architecture = args.architecture
        config.data.processed_dir = str(
            Path(base_config.data.processed_dir) / f"ablation_{variant_name}"
        )
        config.output.runs_dir = str(ablation_dir)

        download_dataset(config.data.raw_dir)
        preprocess_dataset(config)

        trainer = GraphTrainer(config)
        summary = trainer.train()
        rows.append({
            "variant": variant_name,
            "feature_dim": feature_dim(config.features),
            "best_val_mcc": summary["best_val_mcc"],
            "val_mcc": summary["val_metrics"]["mcc"],
            "test_mcc": summary["test_metrics"]["mcc"],
        })

    frame = pd.DataFrame(rows).sort_values("best_val_mcc", ascending=False)
    frame.to_csv(ablation_dir / "feature_ablation.csv", index=False)
    with (ablation_dir / "feature_ablation.json").open("w", encoding="utf-8") as handle:
        json.dump(frame.to_dict(orient="records"), handle, indent=2)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
