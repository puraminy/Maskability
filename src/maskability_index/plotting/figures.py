"""Publication-oriented experiment figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(mi_scores: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """Generate reproduction comparison figures as PNG, PDF, and SVG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    builders = [
        ("mi_vs_evaluation_size", _mi_vs_evaluation_size),
        ("mi_vs_nshot", _mi_vs_nshot),
        ("dr_comparison", _dr_comparison),
        ("model_comparison_mi", lambda df: _model_comparison(df, "maskability_index", "Mean MI")),
        ("model_comparison_dr", lambda df: _model_comparison(df, "dr_prompting", "Mean DR")),
        ("model_comparison_runtime", lambda df: _model_comparison(df, "runtime", "Runtime (s)")),
        ("runtime_comparison", _runtime_comparison),
        ("scatter", _scatter),
        ("heatmap", _heatmap),
    ]
    for name, builder in builders:
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


def _mi_vs_evaluation_size(df: pd.DataFrame) -> None:
    plt.figure(figsize=(5, 3))
    x = "evaluation_size" if "evaluation_size" in df else "sample_size"
    for relation, group in df.groupby("relation"):
        grouped = group.groupby(x)["maskability_index"].mean().sort_index()
        plt.plot(grouped.index, grouped.values, marker="o", label=str(relation))
    plt.xlabel("Evaluation Size")
    plt.ylabel("Maskability Index")
    if df["relation"].nunique() <= 12:
        plt.legend(fontsize="small")


def _mi_vs_nshot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(5, 3))
    x = "few_shot_size" if "few_shot_size" in df else "sample_size"
    for relation, group in df.groupby("relation"):
        grouped = group.groupby(x)["maskability_index"].mean().sort_index()
        plt.plot(grouped.index, grouped.values, marker="o", label=str(relation))
    plt.xlabel("N-shot")
    plt.ylabel("Maskability Index")
    if df["relation"].nunique() <= 12:
        plt.legend(fontsize="small")


def _dr_comparison(df: pd.DataFrame) -> None:
    plt.figure(figsize=(max(5, len(df["relation"].unique()) * 0.6), 3))
    grouped = df.groupby("relation")[["dr_prompting", "dr_masked_prompting"]].mean()
    grouped.plot(kind="bar", ax=plt.gca())
    plt.xlabel("Relation")
    plt.ylabel("DepthRank")


def _model_comparison(df: pd.DataFrame, column: str, ylabel: str) -> None:
    plt.figure(figsize=(5, 3))
    label = "model" if "model" in df else "relation"
    grouped = df.groupby(label)[column].mean().sort_values(ascending=False)
    grouped.plot(kind="bar")
    plt.xlabel(label)
    plt.ylabel(ylabel)


def _runtime_comparison(df: pd.DataFrame) -> None:
    plt.figure(figsize=(5, 3))
    label = "model" if "model" in df else "relation"
    column = "runtime" if "runtime" in df else "runtime_seconds"
    grouped = df.groupby(label)[column].mean().sort_values(ascending=False)
    plt.plot(grouped.index.astype(str), grouped.values, marker="o")
    plt.xlabel(label)
    plt.ylabel("Runtime (s)")
    plt.xticks(rotation=30, ha="right")


def _scatter(df: pd.DataFrame) -> None:
    plt.figure(figsize=(4, 3))
    plt.scatter(df["dr_prompting"], df["dr_masked_prompting"])
    for _, row in df.iterrows():
        plt.annotate(str(row["relation"]), (row["dr_prompting"], row["dr_masked_prompting"]), fontsize=6)
    plt.xlabel("Prompting DR")
    plt.ylabel("Masked DR")


def _heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(5, max(3, df["relation"].nunique() * 0.35)))
    model_col = "model" if "model" in df else "prompt_variant" if "prompt_variant" in df else "relation"
    pivot = df.pivot_table(index="relation", columns=model_col, values="maskability_index", aggfunc="mean")
    plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="MI")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.xlabel(model_col.title())
    plt.ylabel("Relations")
