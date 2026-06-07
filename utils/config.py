import json
from pathlib import Path

import yaml

from schemas import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from a YAML or JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle)
        else:
            payload = json.load(handle)
    return ExperimentConfig.model_validate(payload)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Persist experiment configuration to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.model_dump(), handle, indent=2)
