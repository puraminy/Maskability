"""Unit tests for the manuscript DepthRank equations."""

from __future__ import annotations

import pytest

from maskability_index.depthrank.interfaces import DepthRankResult
from maskability_index.depthrank.ranking import (
    compute_relation_mi,
    maskability_index,
    mean_depthrank,
    relation_depthrank,
)
from maskability_index.depthrank.scoring import depthrank_from_ranks, rank_token


def test_rank_token_implements_index_equation_zero_based() -> None:
    """The highest score has Index zero, matching the paper's rank-zero statement."""
    assert rank_token([0.1, 0.9, 0.3], 1) == 0
    assert rank_token([0.1, 0.9, 0.3], 2) == 1
    assert rank_token([0.1, 0.9, 0.3], 0) == 2


def test_depthrank_from_ranks_implements_full_tail_average() -> None:
    """DepthRank(S) is the arithmetic mean of per-token indices."""
    assert depthrank_from_ranks([1, 53, 1]) == pytest.approx(18.3333333333)
    assert depthrank_from_ranks([33, 4]) == pytest.approx(18.5)


def test_mean_and_relation_depthrank_implement_dr_template_r_n() -> None:
    """Relation-level DR is the mean of instance-level DepthRank values."""
    results = [
        DepthRankResult("p", "t", (1,), ("a",), (2,), 2.0),
        DepthRankResult("p", "t", (2,), ("b",), (4,), 4.0),
    ]
    assert mean_depthrank([2.0, 4.0]) == pytest.approx(3.0)
    assert relation_depthrank(results) == pytest.approx(3.0)


def test_maskability_index_implements_paper_equation() -> None:
    """MI = (DR_Prompting - DR_MaskedPrompting) / DR_Prompting."""
    assert maskability_index(10.0, 7.0) == pytest.approx(0.3)
    assert maskability_index(10.0, 12.0) == pytest.approx(-0.2)


def test_compute_relation_mi_matches_relations_present_in_both_families() -> None:
    """Relation MI is computed from matching prompting and masked samples."""
    prompting = {"r": [DepthRankResult("p", "t", (1,), ("a",), (10,), 10.0)]}
    masked = {"r": [DepthRankResult("p", "t", (1,), ("a",), (7,), 7.0)]}
    assert compute_relation_mi(prompting, masked) == {"r": pytest.approx(0.3)}


def test_empty_target_depthrank_is_undefined() -> None:
    """The equation divides by k, so k=0 must fail."""
    with pytest.raises(ValueError):
        depthrank_from_ranks([])


def test_mi_with_zero_prompting_depthrank_is_undefined() -> None:
    """The MI equation divides by DR_Prompting."""
    with pytest.raises(ZeroDivisionError):
        maskability_index(0.0, 1.0)
