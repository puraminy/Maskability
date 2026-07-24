"""Dataset conversion and tokenization for seq2seq training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from datasets import Dataset, DatasetDict, load_from_disk
from maskability_index.datasets.atomic import RelationInstance
from maskability_index.prompting.builders import PromptBuilder


@dataclass(frozen=True, slots=True)
class TokenizationConfig:
    """Tokenization options for seq2seq fine-tuning examples."""

    max_input_length: int = 128
    max_target_length: int = 32
    cache_dir: str | None = None
    overwrite_cache: bool = False


def instances_to_dataset(
    instances: Sequence[RelationInstance], prompt_builder: PromptBuilder
) -> Dataset:
    """Convert relation instances into a HuggingFace `Dataset` with prompt text and targets."""
    rows = [
        {
            "id": instance.id,
            "head": instance.head,
            "relation": instance.relation,
            "tail": instance.tail,
            "split": instance.split,
            "input_text": prompt_builder.build(instance),
            "target_text": instance.tail,
        }
        for instance in instances
    ]
    return Dataset.from_list(rows)


def instances_to_dataset_dict(
    splits: dict[str, Sequence[RelationInstance]], prompt_builder: PromptBuilder
) -> DatasetDict:
    """Convert split-name keyed relation instances into a HuggingFace `DatasetDict`."""
    datasets = {
        split: instances_to_dataset(instances, prompt_builder)
        for split, instances in splits.items()
    }
    return DatasetDict(datasets)


def tokenize_dataset(
    dataset: Dataset | DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: TokenizationConfig,
) -> Dataset | DatasetDict:
    """Tokenize prompt inputs and target texts, optionally caching the tokenized dataset on disk."""
    if config.cache_dir:
        cache_path = Path(config.cache_dir)
        if cache_path.exists() and not config.overwrite_cache:
            loaded = load_from_disk(str(cache_path))
            if isinstance(dataset, DatasetDict) and not isinstance(loaded, DatasetDict):
                raise TypeError(f"Expected cached DatasetDict at {cache_path}.")
            if isinstance(dataset, Dataset) and not isinstance(loaded, Dataset):
                raise TypeError(f"Expected cached Dataset at {cache_path}.")
            return loaded

    tokenized = dataset.map(
        lambda batch: _tokenize_batch(batch, tokenizer, config),
        batched=True,
        remove_columns=_column_names(dataset),
        desc="Tokenizing seq2seq prompts",
    )

    if config.cache_dir:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tokenized.save_to_disk(str(cache_path))
    return tokenized


def _tokenize_batch(
    batch: dict[str, list[Any]], tokenizer: PreTrainedTokenizerBase, config: TokenizationConfig
) -> dict[str, Any]:
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=config.max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["target_text"], max_length=config.max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def _column_names(dataset: Dataset | DatasetDict) -> list[str]:
    if isinstance(dataset, DatasetDict):
        first_split = next(iter(dataset.keys()))
        return list(dataset[first_split].column_names)
    return list(dataset.column_names)
