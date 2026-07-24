"""Aggregation functions for relation-level DepthRank and Maskability Index."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from maskability_index.depthrank.interfaces import DepthRankResult


def mean_depthrank(values: Sequence[float]) -> float:
    """Compute the arithmetic mean DepthRank for a sample."""
    if not values:
        raise ValueError("Cannot compute a mean DepthRank over an empty sample.")
    return float(sum(values) / len(values))


def relation_depthrank(results: Iterable[DepthRankResult]) -> float:
    """Compute DR_Template(r,n) from per-instance DepthRank results."""
    return mean_depthrank([result.depthrank for result in results])


def maskability_index(dr_prompting: float, dr_masked_prompting: float) -> float:
    """Compute MI(r,n) = (DR_Prompting - DR_MaskedPrompting) / DR_Prompting."""
    if dr_prompting == 0:
        raise ZeroDivisionError("Maskability Index is undefined when DR_Prompting(r,n) is zero.")
    return float((dr_prompting - dr_masked_prompting) / dr_prompting)


def compute_relation_mi(
    prompting: Mapping[str, Sequence[DepthRankResult]],
    masked_prompting: Mapping[str, Sequence[DepthRankResult]],
) -> dict[str, float]:
    """Compute MI for each relation present in both template-family result mappings."""
    relation_names = sorted(set(prompting) & set(masked_prompting))
    return {
        relation: maskability_index(
            relation_depthrank(prompting[relation]), relation_depthrank(masked_prompting[relation])
        )
        for relation in relation_names
    }
