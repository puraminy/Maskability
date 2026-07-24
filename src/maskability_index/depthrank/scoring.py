"""Equation-level scoring functions for DepthRank."""

from __future__ import annotations

from collections.abc import Sequence


def rank_token(scores: Sequence[float], token_id: int, *, zero_based: bool = True) -> int:
    """Implement Index(t_i | h_1:m, r_1:n, t_<i) from the paper.

    Scores are sorted in descending order; the returned value is the sorted-list index of
    `token_id`. The paper's statement that the top prediction has rank zero is represented by
    the default `zero_based=True`.
    """
    if token_id < 0 or token_id >= len(scores):
        raise ValueError(f"token_id {token_id} is outside vocabulary size {len(scores)}.")
    sorted_token_ids = sorted(range(len(scores)), key=lambda index: -scores[index])
    rank = sorted_token_ids.index(token_id)
    return rank if zero_based else rank + 1


def depthrank_from_ranks(token_ranks: Sequence[int]) -> float:
    """Implement DepthRank(S) = (1/k) * sum_i Index(t_i | h, r, t_<i)."""
    if not token_ranks:
        raise ValueError("DepthRank is undefined for an empty target sequence (k = 0).")
    return float(sum(token_ranks) / len(token_ranks))
