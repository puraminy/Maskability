"""DepthRank and Maskability Index public API."""

from maskability_index.depthrank.interfaces import DepthRankResult, DepthRankTokenization
from maskability_index.depthrank.ranking import (
    compute_relation_mi,
    maskability_index,
    mean_depthrank,
    relation_depthrank,
)
from maskability_index.depthrank.scoring import depthrank_from_ranks, rank_token

__all__ = [
    "DepthRankCalculator",
    "DepthRankResult",
    "DepthRankTokenization",
    "compute_relation_mi",
    "depthrank_from_ranks",
    "maskability_index",
    "mean_depthrank",
    "rank_token",
    "relation_depthrank",
]


def __getattr__(name: str) -> object:
    """Lazily import heavy model dependencies only when calculator is requested."""
    if name == "DepthRankCalculator":
        from maskability_index.depthrank.calculator import DepthRankCalculator

        return DepthRankCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
