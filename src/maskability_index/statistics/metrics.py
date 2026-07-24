"""Statistical tests for experiment outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class CorrelationResult:
    """Correlation coefficient and p-value."""

    statistic: float
    pvalue: float


def correlations(x: Sequence[float], y: Sequence[float]) -> dict[str, CorrelationResult]:
    """Compute Pearson, Spearman, and Kendall correlations."""
    _validate_pair(x, y)
    return {
        "pearson": _result(stats.pearsonr(x, y)),
        "spearman": _result(stats.spearmanr(x, y)),
        "kendall": _result(stats.kendalltau(x, y)),
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int = 13,
    iterations: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap a confidence interval around the sample mean."""
    if not values:
        raise ValueError("Bootstrap confidence intervals require at least one value.")
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    means = [
        float(np.mean(rng.choice(sample, size=len(sample), replace=True)))
        for _ in range(iterations)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(sample)),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "confidence": confidence,
    }


def permutation_test(
    x: Sequence[float], y: Sequence[float], *, seed: int = 13, iterations: int = 1000
) -> dict[str, float]:
    """Run a two-sided paired permutation test for a mean difference."""
    _validate_pair(x, y)
    rng = np.random.default_rng(seed)
    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    observed = abs(float(np.mean(diffs)))
    count = 0
    for _ in range(iterations):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        if abs(float(np.mean(diffs * signs))) >= observed:
            count += 1
    return {"statistic": observed, "pvalue": float((count + 1) / (iterations + 1))}


def _result(value: object) -> CorrelationResult:
    return CorrelationResult(float(value.statistic), float(value.pvalue))


def _validate_pair(x: Sequence[float], y: Sequence[float]) -> None:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Paired statistics require equal-length samples with at least two values.")
