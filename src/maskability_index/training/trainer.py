"""Reusable HuggingFace seq2seq trainer wrapper."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from datasets import Dataset
from maskability_index.training.metrics import build_compute_metrics


@dataclass(frozen=True, slots=True)
class TrainingPipelineConfig:
    """Hydra-exposed seq2seq training hyperparameters."""

    output_dir: str = "results/checkpoints"
    epochs: float = 3.0
    batch_size: int = 8
    learning_rate: float = 5e-5
    optimizer: str = "adamw_torch"
    scheduler: str = "linear"
    warmup: int = 0
    weight_decay: float = 0.0
    generation_length: int = 32
    beam_size: int = 1
    seed: int = 13
    mixed_precision: bool = False
    logging_steps: int = 50
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    predict_with_generate: bool = True


class MaskabilitySeq2SeqTrainer:
    """Thin deterministic wrapper around HuggingFace `Seq2SeqTrainer`."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset | None,
        eval_dataset: Dataset | None,
        config: TrainingPipelineConfig,
    ) -> None:
        """Initialize the underlying HuggingFace trainer."""
        self.config = config
        set_seed(config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        argument_kwargs = {
            "output_dir": config.output_dir,
            "num_train_epochs": config.epochs,
            "per_device_train_batch_size": config.batch_size,
            "per_device_eval_batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "optim": config.optimizer,
            "lr_scheduler_type": config.scheduler,
            "warmup_steps": config.warmup,
            "weight_decay": config.weight_decay,
            "predict_with_generate": config.predict_with_generate,
            "generation_max_length": config.generation_length,
            "generation_num_beams": config.beam_size,
            "seed": config.seed,
            "data_seed": config.seed,
            "fp16": config.mixed_precision,
            "logging_steps": config.logging_steps,
            "save_strategy": config.save_strategy,

            "report_to": [],
        }
        strategy_key = "evaluation_strategy"
        if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments).parameters:
            strategy_key = "eval_strategy"
        argument_kwargs[strategy_key] = config.evaluation_strategy
        args = Seq2SeqTrainingArguments(**argument_kwargs)
        trainer_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": data_collator,
            "compute_metrics": build_compute_metrics(
                tokenizer if config.predict_with_generate else None
            ),
        }
        tokenizer_key = "tokenizer"
        if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters:
            tokenizer_key = "processing_class"
        trainer_kwargs[tokenizer_key] = tokenizer
        self.trainer = Seq2SeqTrainer(**trainer_kwargs)

    def train(self, resume_from_checkpoint: str | Path | bool | None = None) -> Any:
        """Train, optionally resuming from a checkpoint path or latest checkpoint."""
        return self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def evaluate(self, eval_dataset: Dataset | None = None) -> dict[str, float]:
        """Evaluate and return validation metrics including validation loss."""
        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
        return {str(k): float(v) for k, v in metrics.items() if isinstance(v, int | float)}

    def predict(self, test_dataset: Dataset) -> Any:
        """Generate predictions for a tokenized test dataset."""
        return self.trainer.predict(test_dataset=test_dataset)

    def save_checkpoint(self, output_dir: str | Path | None = None) -> None:
        """Save model, tokenizer, and trainer state."""
        destination = str(output_dir or self.config.output_dir)
        self.trainer.save_model(destination)
        self.trainer.save_state()

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Load model weights from a checkpoint directory into the current trainer model."""
        loaded = type(self.trainer.model).from_pretrained(str(checkpoint_dir))
        self.trainer.model = loaded
