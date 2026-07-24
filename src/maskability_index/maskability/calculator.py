"""Public calculator for the Maskability Index."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from maskability_index.depthrank import DepthRankResult
from maskability_index.maskability.aggregation import mean_depthrank, relation_depthrank
from maskability_index.maskability.interfaces import MaskabilityResult
from maskability_index.maskability.metrics import classify_maskability, maskability_index


@dataclass(frozen=True, slots=True)
class MaskabilityCalculator:
    """Compute MI exactly as the manuscript defines it."""

    threshold: float | None = None

    def compute(
        self,
        relation: str,
        prompting: Sequence[float] | Sequence[DepthRankResult],
        masked_prompting: Sequence[float] | Sequence[DepthRankResult],
        sample_size: int | None = None,
    ) -> MaskabilityResult:
        """Compute MI(r,n) from prefix and masked DepthRank samples."""
        dr_prompting = _mean_depthrank_sample(prompting)
        dr_masked = _mean_depthrank_sample(masked_prompting)
        n = sample_size if sample_size is not None else min(len(prompting), len(masked_prompting))
        mi = maskability_index(dr_prompting, dr_masked)
        group = classify_maskability(mi, self.threshold) if self.threshold is not None else None
        return MaskabilityResult(relation, n, dr_prompting, dr_masked, mi, group)

    def compute_many(
        self,
        prompting: Mapping[str, Sequence[float] | Sequence[DepthRankResult]],
        masked_prompting: Mapping[str, Sequence[float] | Sequence[DepthRankResult]],
    ) -> dict[str, MaskabilityResult]:
        """Compute MI for relations available in both template-family mappings."""
        return {
            relation: self.compute(relation, prompting[relation], masked_prompting[relation])
            for relation in sorted(set(prompting) & set(masked_prompting))
        }


def _mean_depthrank_sample(values: Sequence[float] | Sequence[DepthRankResult]) -> float:
    if not values:
        raise ValueError("MI requires non-empty DepthRank samples.")
    first = values[0]
    if isinstance(first, DepthRankResult):
        return relation_depthrank(values)  # type: ignore[arg-type]
    return mean_depthrank(values)  # type: ignore[arg-type]
