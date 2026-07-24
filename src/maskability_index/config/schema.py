"""Typed configuration schema for Maskability experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset acquisition and split configuration."""

    name: str
    hf_path: str
    cache_dir: str
    splits: list[str] = field(default_factory=lambda: ["train", "validation", "test"])


@dataclass(frozen=True)
class ModelConfig:
    """HuggingFace model and tokenizer configuration."""

    name: str
    tokenizer_name: str
    revision: str = "main"


@dataclass(frozen=True)
class PromptingConfig:
    """Prompt construction configuration without hardcoded scientific templates."""

    style: str
    n_shot: int
    template_set: str


@dataclass(frozen=True)
class TrackingConfig:
    """Experiment tracking backend configuration."""

    backend: str = "local"
    mlflow_tracking_uri: str | None = None
    experiment_name: str = "default"


@dataclass(frozen=True)
class OutputConfig:
    """Output directory layout requested for every experiment."""

    create_subdirs: list[str] = field(default_factory=lambda: ["logs", "checkpoints", "plots", "latex"])


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
