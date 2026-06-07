import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from data.splits import load_splits


class RedditThreadsDataset(Dataset):
    """Lazy sharded dataset for Reddit Threads graphs."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
    ) -> None:
        processed_dir = Path(processed_dir)
        manifest_path = processed_dir / "manifest.json"
        splits_path = processed_dir / "splits.json"

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.shard_dir = processed_dir / "shards"
        self.shard_names = manifest["shards"]
        self.graph_index = manifest["graph_index"]
        self.split_indices = load_splits(splits_path)[split]
        self._shard_cache: dict[int, list[Data]] = {}

    def _load_shard(self, shard_id: int) -> list[Data]:
        if shard_id not in self._shard_cache:
            shard_name = self.shard_names[shard_id]
            self._shard_cache = {
                shard_id: torch.load(
                    self.shard_dir / shard_name,
                    weights_only=False,
                )
            }
        return self._shard_cache[shard_id]

    def __len__(self) -> int:
        return self.split_indices.shape[0]

    def __getitem__(self, index: int) -> Data:
        global_idx = int(self.split_indices[index])
        shard_id, local_idx = self.graph_index[global_idx]
        shard = self._load_shard(shard_id)
        return shard[local_idx]


def collate_graphs(batch: list[Data]) -> Batch:
    """Collate a list of graphs into a PyG batch."""
    return Batch.from_data_list(batch)


def load_split_indices(processed_dir: str | Path) -> dict[str, np.ndarray]:
    """Load train, validation, and test split indices."""
    return load_splits(Path(processed_dir) / "splits.json")
