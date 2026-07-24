"""Typed interfaces for Maskability Index computations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MaskabilityResult:
    """Maskability Index output for one relation and sample-size setting."""

    relation: str
    sample_size: int
    dr_prompting: float
    dr_masked_prompting: float
    maskability_index: float
    group: str | None = None
