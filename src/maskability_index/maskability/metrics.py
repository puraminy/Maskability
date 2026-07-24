"""Equation-level Maskability Index metrics."""

from __future__ import annotations


def maskability_index(dr_prompting: float, dr_masked_prompting: float) -> float:
    """Compute MI(r,n) = (DR_Prompting(r,n) - DR_MaskedPrompting(r,n)) / DR_Prompting(r,n)."""
    if dr_prompting == 0:
        raise ZeroDivisionError("MI is undefined when DR_Prompting(r,n) is zero.")
    return float((dr_prompting - dr_masked_prompting) / dr_prompting)


def classify_maskability(mi: float, threshold: float = 0.30) -> str:
    """Apply the manuscript's experimental grouping rule."""
    return "Mask-Filling" if mi >= threshold else "Map-Phrasal"
