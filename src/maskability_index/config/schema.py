"""Typed configuration schema for Maskability experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSamplingConfig:
    """Relation-balanced evaluation sampling configuration."""

    strategy: str = "deterministic"
    instances_per_relation: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset acquisition, split, and sampling configuration."""

    name: str
    hf_path: str
    cache_dir: str
    backend: str = "auto"
    local_path: str = "data/atomic"
    split: str = "validation"
    splits: list[str] = field(default_factory=lambda: ["train", "validation", "test"])
    sampling: DatasetSamplingConfig = field(default_factory=DatasetSamplingConfig)


@dataclass(frozen=True)
class ModelConfig:
    """HuggingFace model and tokenizer configuration."""

    name: str
    tokenizer_name: str
    revision: str = "main"


@dataclass(frozen=True)
class PromptingConfig:
    """Prompt construction configuration independent of evaluation sampling."""

    style: str
    n_shot: int
    template_set: str


@dataclass(frozen=True)
class TrainingConfig:
    """Seq2Seq training pipeline configuration."""

    epochs: float = 3.0
    batch_size: int = 8
    learning_rate: float = 5e-5
    optimizer: str = "adamw_torch"
    scheduler: str = "linear"
    warmup: int = 0
    weight_decay: float = 0.0
    max_input_length: int = 128
    max_target_length: int = 32
    generation_length: int = 32
    beam_size: int = 1
    seed: int = 13
    mixed_precision: bool = False
    output_dir: str = "results/checkpoints"
    tokenized_cache_dir: str | None = None
    overwrite_cache: bool = False


@dataclass(frozen=True)
class TrackingConfig:
    """Experiment tracking backend configuration."""

    backend: str = "local"
    mlflow_tracking_uri: str | None = None
    experiment_name: str = "default"


@dataclass(frozen=True)
class OutputConfig:
    """Output directory layout requested for every experiment."""

    create_subdirs: list[str] = field(
        default_factory=lambda: ["logs", "checkpoints", "plots", "latex"]
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment options loaded by Hydra."""

    name: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    prompting: PromptingConfig
    tracking: TrackingConfig
    outputs: OutputConfig
    training: TrainingConfig | None = None
