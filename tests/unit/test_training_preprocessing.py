"""Tests for seq2seq training preprocessing."""

from datasets import Dataset
from maskability_index.datasets import RelationInstance
from maskability_index.prompting import MaskedPromptBuilder, PrefixPromptBuilder
from maskability_index.training.preprocessing import (
    TokenizationConfig,
    instances_to_dataset,
    tokenize_dataset,
)


class DummyTokenizer:
    """Minimal tokenizer stub for preprocessing tests."""

    pad_token_id = 0

    def __call__(self, text=None, text_target=None, max_length=8, truncation=True):
        """Tokenize text by whitespace token length."""
        values = text_target if text_target is not None else text
        ids = [[min(len(token), 99) for token in value.split()][:max_length] for value in values]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


def test_instances_to_dataset_uses_prefix_prompt() -> None:
    """Dataset conversion should store prefix prompts and tail targets."""
    instance = RelationInstance("PersonX cooks", "xNeed", "buy food", "train", "1")
    dataset = instances_to_dataset([instance], PrefixPromptBuilder())
    assert isinstance(dataset, Dataset)
    assert dataset[0]["target_text"] == "buy food"
    assert "buy food" not in dataset[0]["input_text"]


def test_instances_to_dataset_supports_masked_prompt() -> None:
    """Dataset conversion should support masked prompts through prompt builders."""
    instance = RelationInstance("PersonX cooks", "xNeed", "buy food", "train", "1")
    dataset = instances_to_dataset([instance], MaskedPromptBuilder())
    assert "<extra_id_0>" in dataset[0]["input_text"]


def test_tokenize_dataset_adds_labels() -> None:
    """Tokenization should produce model inputs and target labels."""
    instance = RelationInstance("PersonX cooks", "xNeed", "buy food", "train", "1")
    dataset = instances_to_dataset([instance], PrefixPromptBuilder())
    tokenized = tokenize_dataset(
        dataset, DummyTokenizer(), TokenizationConfig(max_input_length=4, max_target_length=3)
    )
    assert set(tokenized.column_names) == {"input_ids", "attention_mask", "labels"}
    assert tokenized[0]["labels"]
