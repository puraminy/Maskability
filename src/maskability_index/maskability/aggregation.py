"""Aggregation functions used by the Maskability Index equation."""

from __future__ import annotations

from collections.abc import Sequence

from maskability_index.depthrank import DepthRankResult
from maskability_index.depthrank import relation_depthrank as depthrank_relation_depthrank


def mean_depthrank(values: Sequence[float]) -> float:
    """Compute relation-level mean DepthRank as an arithmetic mean."""
    if not values:
        raise ValueError("Cannot compute mean DepthRank over an empty sample.")
    return float(sum(values) / len(values))


def relation_depthrank(results: Sequence[DepthRankResult]) -> float:
    """Compute DR_Template(r,n) from public DepthRank results."""
    if not results:
        raise ValueError("Cannot compute relation DepthRank over an empty sample.")
    return float(depthrank_relation_depthrank(results))
