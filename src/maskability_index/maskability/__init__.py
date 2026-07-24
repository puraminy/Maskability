"""Maskability Index public API."""

from maskability_index.maskability.aggregation import mean_depthrank, relation_depthrank
from maskability_index.maskability.calculator import MaskabilityCalculator
from maskability_index.maskability.interfaces import MaskabilityResult
from maskability_index.maskability.metrics import classify_maskability, maskability_index

__all__ = [
    "MaskabilityCalculator",
    "MaskabilityResult",
    "classify_maskability",
    "maskability_index",
    "mean_depthrank",
    "relation_depthrank",
]
