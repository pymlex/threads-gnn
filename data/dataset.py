import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm.auto import tqdm

from data.splits import load_splits


class RedditThreadsDataset(Dataset):
    """Dataset for Reddit Threads graphs with split-level shard preloading."""

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
        self._graphs = self._preload_split(split)

    def _preload_split(self, split: str) -> list[Data]:
        """Load all shards required by a split once into memory."""
        shard_ids = {
            self.graph_index[int(global_idx)][0]
            for global_idx in self.split_indices
        }
        shard_buffers: dict[int, list[Data]] = {}
        for shard_id in tqdm(sorted(shard_ids), desc=f"Loading {split} shards"):
            shard_name = self.shard_names[shard_id]
            shard_buffers[shard_id] = torch.load(
                self.shard_dir / shard_name,
                weights_only=False,
            )
        graphs = []
        for global_idx in self.split_indices:
            shard_id, local_idx = self.graph_index[int(global_idx)]
            graphs.append(shard_buffers[shard_id][local_idx])
        return graphs

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index: int) -> Data:
        return self._graphs[index]


def collate_graphs(batch: list[Data]) -> Batch:
    """Collate a list of graphs into a PyG batch."""
    return Batch.from_data_list(batch)


def load_split_indices(processed_dir: str | Path) -> dict[str, np.ndarray]:
    """Load train, validation, and test split indices."""
    return load_splits(Path(processed_dir) / "splits.json")
