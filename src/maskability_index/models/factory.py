"""Extensible model factory for supported seq2seq language models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass(frozen=True, slots=True)
class Seq2SeqModelBundle:
    """Loaded seq2seq model and tokenizer pair."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


@dataclass(frozen=True, slots=True)
class Seq2SeqModelSpec:
    """Factory specification for one seq2seq model family member."""

    model_loader: Callable[[str, str], PreTrainedModel]
    tokenizer_loader: Callable[[str, str], PreTrainedTokenizerBase]
    tokenizer_name: str | None = None


SUPPORTED_SEQ2SEQ_MODELS: dict[str, Seq2SeqModelSpec] = {
    "google-t5/t5-small": Seq2SeqModelSpec(
        AutoModelForSeq2SeqLM.from_pretrained, AutoTokenizer.from_pretrained
    ),
    "google-t5/t5-base": Seq2SeqModelSpec(
        AutoModelForSeq2SeqLM.from_pretrained, AutoTokenizer.from_pretrained
    ),
    "google-t5/t5-large": Seq2SeqModelSpec(
        AutoModelForSeq2SeqLM.from_pretrained, AutoTokenizer.from_pretrained
    ),
}


class Seq2SeqModelFactory:
    """Registry-backed loader so training code is independent of model names."""

    def __init__(self, registry: dict[str, Seq2SeqModelSpec] | None = None) -> None:
        """Initialize the factory with a copy of the supplied registry."""
        self._registry = dict(registry or SUPPORTED_SEQ2SEQ_MODELS)

    def register(self, name: str, spec: Seq2SeqModelSpec) -> None:
        """Register or replace a seq2seq model specification."""
        self._registry[name] = spec

    def create(
        self, name: str, revision: str = "main", tokenizer_name: str | None = None
    ) -> Seq2SeqModelBundle:
        """Load a supported model and tokenizer bundle."""
        if name not in self._registry:
            supported = ", ".join(sorted(self._registry))
            raise ValueError(f"Unsupported seq2seq model {name!r}. Supported models: {supported}.")
        spec = self._registry[name]
        resolved_tokenizer = tokenizer_name or spec.tokenizer_name or name
        tokenizer = spec.tokenizer_loader(resolved_tokenizer, revision=revision)
        model = spec.model_loader(name, revision=revision)
        return Seq2SeqModelBundle(model=model, tokenizer=tokenizer)


def create_seq2seq_model(
    name: str, revision: str = "main", tokenizer_name: str | None = None
) -> Seq2SeqModelBundle:
    """Create a seq2seq model/tokenizer bundle from the default factory."""
    return Seq2SeqModelFactory().create(
        name=name, revision=revision, tokenizer_name=tokenizer_name
    )
