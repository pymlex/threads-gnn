from typing import Literal

from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    use_degree: bool = True
    use_log_degree: bool = True
    use_normalised_degree: bool = True
    use_degree_bucket: bool = True
    degree_bucket_bins: int = 16
    use_clustering: bool = True
    use_kcore: bool = True
    use_pagerank: bool = True
    use_laplacian_pe: bool = True
    laplacian_pe_dim: int = 8
    use_rwse: bool = True
    rwse_steps: int = 8
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 100


class ModelConfig(BaseModel):
    architecture: Literal["gin", "pna", "gat"] = "gin"
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    num_heads: int = 4
    use_virtual_node: bool = True
    pooling: Literal["mean", "sum", "attention"] = "attention"
    gin_eps: float = 0.0
    pna_aggregators: list[str] = Field(
        default_factory=lambda: ["mean", "max", "min", "std"]
    )
    pna_scalers: list[str] = Field(
        default_factory=lambda: ["identity", "amplification", "attenuation"]
    )


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    shard_size: int = 2000
    preprocess_workers: int = 8
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


class TrainingConfig(BaseModel):
    batch_size: int = 512
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stopping_patience: int = 20
    scheduler: Literal["cosine", "plateau"] = "cosine"
    use_amp: bool = True
    num_workers: int = 0


class OutputConfig(BaseModel):
    runs_dir: str = "runs"
    checkpoints_dir: str = "checkpoints"


class ExperimentConfig(BaseModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
