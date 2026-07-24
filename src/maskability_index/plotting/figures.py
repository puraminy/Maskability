"""Publication-oriented experiment figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(mi_scores: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """Generate standard publication figures as PNG, PDF, and SVG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, builder in [
        ("scatter", _scatter),
        ("histogram", _histogram),
        ("correlation", _correlation),
        ("threshold", _threshold),
        ("sensitivity", _sensitivity),
        ("model_comparison", _model_comparison),
        ("baseline_comparison", _baseline_comparison),
    ]:
        builder(mi_scores)
        paths.extend(_save_all(out / name))
    return paths


def _save_all(stem: Path) -> list[Path]:
    plt.tight_layout()
    paths = [stem.with_suffix(suffix) for suffix in [".png", ".pdf", ".svg"]]
    for path in paths:
        plt.savefig(path)
    plt.close()
    return paths


def _scatter(df: pd.DataFrame) -> None:
    plt.figure(figsize=(4, 3))
    plt.scatter(df["dr_prompting"], df["dr_masked_prompting"])
    plt.xlabel("DR Prompting")
    plt.ylabel("DR MaskedPrompting")


def _histogram(df: pd.DataFrame) -> None:
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
    return None


def _correlation(df: pd.DataFrame) -> None:
    plt.figure(figsize=(4, 3))
    plt.scatter(df["sample_size"], df["maskability_index"])
    plt.xlabel("n-shot / sample size")
    plt.ylabel("MI")
    return None


def _threshold(df: pd.DataFrame) -> None:
    plt.figure(figsize=(4, 3))
    threshold = float(df.attrs.get("threshold", 0.30))
    plt.axhline(threshold, linestyle="--")
    plt.plot(range(len(df)), df["maskability_index"], marker="o")
    plt.xlabel("Relation index")
    plt.ylabel("MI")
    return None


def _sensitivity(df: pd.DataFrame) -> None:
    plt.figure(figsize=(4, 3))
    grouped = df.groupby("sample_size")["maskability_index"].mean()
    plt.plot(grouped.index, grouped.values, marker="o")
    plt.xlabel("Sample size")
    plt.ylabel("Mean MI")
    return None


def _model_comparison(df: pd.DataFrame) -> None:
    plt.figure(figsize=(4, 3))
    label = "model" if "model" in df else "run" if "run" in df else "relation"
    grouped = df.groupby(label)["maskability_index"].mean().sort_values(ascending=False)
    grouped.plot(kind="bar")
    plt.xlabel(label)
    plt.ylabel("Mean MI")


def _baseline_comparison(df: pd.DataFrame) -> None:
    plt.figure(figsize=(4, 3))
    label = "baseline" if "baseline" in df else "group" if "group" in df else "relation"
    grouped = df.groupby(label)["maskability_index"].mean().sort_values(ascending=False)
    grouped.plot(kind="bar")
    plt.xlabel(label)
    plt.ylabel("Mean MI")
