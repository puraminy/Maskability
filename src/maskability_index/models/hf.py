"""HuggingFace Transformers factories."""

from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_tokenizer(name: str, revision: str = "main") -> PreTrainedTokenizerBase:
    """Load a tokenizer from HuggingFace Transformers."""

    return AutoTokenizer.from_pretrained(name, revision=revision)


def load_seq2seq_model(name: str, revision: str = "main") -> PreTrainedModel:
    """Load a sequence-to-sequence model from HuggingFace Transformers."""

    return AutoModelForSeq2SeqLM.from_pretrained(name, revision=revision)
