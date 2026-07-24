"""Publication-oriented experiment figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(mi_scores: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """Generate standard PDF plots for MI experiments."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        _scatter(mi_scores, out / "scatter.pdf"),
        _histogram(mi_scores, out / "histogram.pdf"),
        _correlation(mi_scores, out / "correlation.pdf"),
        _threshold(mi_scores, out / "threshold.pdf"),
        _sensitivity(mi_scores, out / "sensitivity.pdf"),
    ]


def _save(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()
    return path


def _scatter(df: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(4, 3))
    plt.scatter(df["dr_prompting"], df["dr_masked_prompting"])
    plt.xlabel("DR Prompting")
    plt.ylabel("DR MaskedPrompting")
    return _save(path)


def _histogram(df: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(4, 3))
    values = df["maskability_index"]
    value_range = abs(float(values.max()) - float(values.min()))
    bins = 1 if value_range < 1e-12 else min(10, max(1, len(df)))
    hist_range = None
    if bins == 1:
        hist_range = (float(values.min()) - 0.5, float(values.max()) + 0.5)
    plt.hist(values, bins=bins, range=hist_range)
    plt.xlabel("Maskability Index")
    plt.ylabel("Relations")
    return _save(path)


def _correlation(df: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(4, 3))
    plt.scatter(df["sample_size"], df["maskability_index"])
    plt.xlabel("n-shot / sample size")
    plt.ylabel("MI")
    return _save(path)


def _threshold(df: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(4, 3))
    threshold = float(df.attrs.get("threshold", 0.30))
    plt.axhline(threshold, linestyle="--")
    plt.plot(range(len(df)), df["maskability_index"], marker="o")
    plt.xlabel("Relation index")
    plt.ylabel("MI")
    return _save(path)


def _sensitivity(df: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(4, 3))
    grouped = df.groupby("sample_size")["maskability_index"].mean()
    plt.plot(grouped.index, grouped.values, marker="o")
    plt.xlabel("Sample size")
    plt.ylabel("Mean MI")
    return _save(path)
