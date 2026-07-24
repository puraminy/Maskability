"""Unit tests for Maskability Index computations."""

from __future__ import annotations

import pytest

from maskability_index.depthrank import DepthRankResult
from maskability_index.maskability import (
    MaskabilityCalculator,
    classify_maskability,
    maskability_index,
    mean_depthrank,
    relation_depthrank,
)


def test_maskability_index_equation_toy_example() -> None:
    """MI is the manuscript's relative DepthRank improvement equation."""
    assert maskability_index(10.0, 7.0) == pytest.approx(0.30)


def test_calculator_computes_means_and_group_deterministically() -> None:
    """Calculator uses arithmetic means and the configured grouping threshold."""
    calculator = MaskabilityCalculator(threshold=0.30)
    first = calculator.compute("AtLocation", [10.0, 14.0], [6.0, 8.0])
    second = calculator.compute("AtLocation", [10.0, 14.0], [6.0, 8.0])
    assert first == second
    assert first.sample_size == 2
    assert first.dr_prompting == pytest.approx(12.0)
    assert first.dr_masked_prompting == pytest.approx(7.0)
    assert first.maskability_index == pytest.approx(5.0 / 12.0)
    assert first.group == "Mask-Filling"


def test_negative_mi_classifies_as_map_phrasal() -> None:
    """Negative MI means prefix prompting is relatively better."""
    result = MaskabilityCalculator(threshold=0.30).compute("xIntent", [10.0], [16.0])
    assert result.maskability_index == pytest.approx(-0.6)
    assert result.group == "Map-Phrasal"


def test_depthrank_result_inputs_use_public_depthrank_result() -> None:
    """MI can aggregate public DepthRankResult objects without DepthRank code duplication."""
    prompting = [DepthRankResult("p", "t", (1,), ("t",), (10,), 10.0)]
    masked = [DepthRankResult("m", "t", (1,), ("t",), (5,), 5.0)]
    assert relation_depthrank(prompting) == pytest.approx(10.0)
    mi = MaskabilityCalculator().compute("rel", prompting, masked).maskability_index
    assert mi == pytest.approx(0.5)


def test_edge_cases() -> None:
    """Empty samples and zero denominators are rejected."""
    with pytest.raises(ValueError):
        mean_depthrank([])
    with pytest.raises(ZeroDivisionError):
        maskability_index(0.0, 1.0)


def test_threshold_boundary() -> None:
    """The paper's threshold rule is inclusive at 0.30."""
    assert classify_maskability(0.30) == "Mask-Filling"
    assert classify_maskability(0.299) == "Map-Phrasal"
