"""Training utilities for seq2seq language-model fine-tuning."""

from maskability_index.training.metrics import GenerationMetrics, build_compute_metrics
from maskability_index.training.preprocessing import (
    TokenizationConfig,
    instances_to_dataset,
    instances_to_dataset_dict,
    tokenize_dataset,
)
from maskability_index.training.trainer import MaskabilitySeq2SeqTrainer, TrainingPipelineConfig

__all__ = [
    "GenerationMetrics",
    "MaskabilitySeq2SeqTrainer",
    "TokenizationConfig",
    "TrainingPipelineConfig",
    "build_compute_metrics",
    "instances_to_dataset",
    "instances_to_dataset_dict",
    "tokenize_dataset",
]
