"""Typed interfaces and results for DepthRank computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DepthRankTokenization:
    """Tokenized representation of S = {h_1:m, r_1:n, t_1:k}."""

    prompt_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    target_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DepthRankResult:
    """DepthRank output for one prompt and one gold target tail."""

    prompt: str
    target: str
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    token_ranks: tuple[int, ...]
    depthrank: float


class DepthRankModel(Protocol):
    """Minimal callable model protocol returning teacher-forced logits."""

    def __call__(self, **kwargs: object) -> object:
        """Return an object with a `logits` tensor of shape batch x target x vocab."""
