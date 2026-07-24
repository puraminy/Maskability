"""Train a seq2seq language model without DepthRank or Maskability Index calculations."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from datasets import DatasetDict
from maskability_index.datasets.atomic import load_atomic2020_instances
from maskability_index.models.factory import create_seq2seq_model
from maskability_index.prompting.builders import builder_from_style
from maskability_index.training.preprocessing import (
    TokenizationConfig,
    instances_to_dataset_dict,
    tokenize_dataset,
)
from maskability_index.training.trainer import MaskabilitySeq2SeqTrainer, TrainingPipelineConfig
from maskability_index.utils.reproducibility import set_seed


def _training_config(cfg: DictConfig) -> TrainingPipelineConfig:
    training = cfg.experiment.training
    return TrainingPipelineConfig(
        output_dir=str(training.output_dir),
        epochs=float(training.epochs),
        batch_size=int(training.batch_size),
        learning_rate=float(training.learning_rate),
        optimizer=str(training.optimizer),
        scheduler=str(training.scheduler),
        warmup=int(training.warmup),
        weight_decay=float(training.weight_decay),
        generation_length=int(training.generation_length),
        beam_size=int(training.beam_size),
        seed=int(training.seed),
        mixed_precision=bool(training.mixed_precision),
        logging_steps=int(training.logging_steps),
        save_strategy=str(training.save_strategy),
        evaluation_strategy=str(training.evaluation_strategy),
        predict_with_generate=bool(training.predict_with_generate),
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load data, tokenize prompts, train, evaluate, and save a seq2seq checkpoint."""
    set_seed(int(cfg.experiment.seed))
    model_bundle = create_seq2seq_model(
        name=str(cfg.experiment.model.name),
        revision=str(cfg.experiment.model.revision),
        tokenizer_name=str(cfg.experiment.model.tokenizer_name),
    )
    prompt_builder = builder_from_style(str(cfg.experiment.prompting.style))
    split_names = [str(split) for split in cfg.experiment.dataset.splits]
    instances = {
        split: load_atomic2020_instances(
            split=split,
            cache_dir=str(cfg.experiment.dataset.cache_dir),
            hf_path=str(cfg.experiment.dataset.hf_path),
        )
        for split in split_names
    }
    raw_datasets = instances_to_dataset_dict(instances, prompt_builder)
    tokenized = tokenize_dataset(
        raw_datasets,
        model_bundle.tokenizer,
        TokenizationConfig(
            max_input_length=int(cfg.experiment.training.max_input_length),
            max_target_length=int(cfg.experiment.training.max_target_length),
            cache_dir=str(cfg.experiment.training.tokenized_cache_dir),
            overwrite_cache=bool(cfg.experiment.training.overwrite_cache),
        ),
    )
    if not isinstance(tokenized, DatasetDict):
        raise TypeError("Training script requires a tokenized DatasetDict.")
    output_dir = Path(str(cfg.experiment.training.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    trainer = MaskabilitySeq2SeqTrainer(
        model=model_bundle.model,
        tokenizer=model_bundle.tokenizer,
        train_dataset=tokenized.get("train"),
        eval_dataset=tokenized.get("validation") or tokenized.get("dev"),
        config=_training_config(cfg),
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_checkpoint(output_dir)


if __name__ == "__main__":
    main()
