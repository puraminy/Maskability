"""Typed configuration schema for Maskability experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSamplingConfig:
    """Legacy raw-instance sampling configuration; prefer few_shot/evaluation."""

    strategy: str = "deterministic"
    instances_per_relation: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class RelationsConfig:
    """Relation selection configuration."""

    mode: str = "selected"
    selected: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DepthRankEvaluationConfig:
    """Held-out DepthRank evaluation sampling configuration."""

    heads_per_relation: int | None = 100
    max_reference_tails: int | None = 3
    strategy: str = "deterministic"
    seed: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """Independent evaluation-size configuration."""

    max_instances_per_relation: int | str = "all"
    depthrank: DepthRankEvaluationConfig = field(default_factory=DepthRankEvaluationConfig)


@dataclass(frozen=True)
class FewShotConfig:
    """Few-shot training/adaptation or demonstration configuration."""

    enabled: bool = False
    n_samples: int = 0
    strategy: str = "deterministic"
    seed: int | None = None


@dataclass(frozen=True)
class DemonstrationsConfig:
    """Few-shot demonstration configuration independent of evaluation size."""

    enabled: bool = False
    num_examples: int = 0
    strategy: str = "deterministic"
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
    demonstrations: DemonstrationsConfig = field(default_factory=DemonstrationsConfig)


@dataclass(frozen=True)
class SweepConfig:
    """Generic experiment sweep configuration."""

    enabled: bool = False
    dimensions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisConfig:
    """Statistical analysis and sweep values."""

    threshold: float = 0.30
    bootstrap_iterations: int = 1000
    permutation_iterations: int = 1000
    thresholds: list[float] = field(default_factory=list)
    instances_per_relation: list[int] = field(default_factory=list)
    n_shots: list[int] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    prompt_variants: list[str] = field(default_factory=list)


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
    relations: RelationsConfig
    evaluation: EvaluationConfig
    model: ModelConfig
    prompting: PromptingConfig
    few_shot: FewShotConfig
    sweep: SweepConfig
    analysis: AnalysisConfig
    tracking: TrackingConfig
    outputs: OutputConfig
    training: TrainingConfig | None = None
