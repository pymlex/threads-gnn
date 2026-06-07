import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from schemas import DataConfig


def create_stratified_splits(
    labels: np.ndarray,
    config: DataConfig,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create stratified train, validation, and test index splits."""
    indices = np.arange(labels.shape[0])
    train_ratio = config.train_ratio
    val_ratio = config.val_ratio
    test_ratio = config.test_ratio
    val_fraction = val_ratio / (train_ratio + val_ratio)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=labels[temp_idx],
        random_state=seed,
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def save_splits(
    splits: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    """Persist split indices to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {key: value.tolist() for key, value in splits.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2)


def load_splits(path: str | Path) -> dict[str, np.ndarray]:
    """Load split indices from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: np.array(value, dtype=np.int64) for key, value in payload.items()}


def build_splits_from_targets(
    target_csv: str | Path,
    config: DataConfig,
    seed: int,
    output_path: str | Path,
) -> dict[str, np.ndarray]:
    """Build and save stratified splits from reddit_target.csv."""
    labels = pd.read_csv(target_csv)["target"].to_numpy(dtype=np.int64)
    splits = create_stratified_splits(labels, config, seed)
    save_splits(splits, output_path)
    for split_name, split_idx in splits.items():
        tqdm.write(
            f"{split_name}: {split_idx.shape[0]} graphs, "
            f"positive rate {labels[split_idx].mean():.4f}"
        )
    return splits
