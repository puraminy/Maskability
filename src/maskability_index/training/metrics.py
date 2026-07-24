"""Metrics for seq2seq training and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Callable metric computer designed for extension with future metrics."""

    tokenizer: PreTrainedTokenizerBase | None = None

    def __call__(self, eval_prediction: Any) -> dict[str, float]:
        """Compute exact-match generation accuracy when predictions and labels are available."""
        if self.tokenizer is None:
            return {}
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        labels = np.where(labels == -100, self.tokenizer.pad_token_id, labels)
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        matches = [
            prediction.strip() == label.strip()
            for prediction, label in zip(decoded_predictions, decoded_labels, strict=False)
        ]
        return {"generation_accuracy": float(np.mean(matches)) if matches else 0.0}


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase | None) -> GenerationMetrics | None:
    """Return a metric callable for `Seq2SeqTrainer`, or `None` when generation is disabled."""
    if tokenizer is None:
        return None
    return GenerationMetrics(tokenizer)
