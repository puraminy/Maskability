"""Comparison utilities for completed experiment runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from maskability_index.evaluation.evaluator import aggregate_metrics, load_experiment


def compare_runs(run_dirs: list[Path | str]) -> pd.DataFrame:
    """Compare completed runs and rank configurations by mean Maskability Index."""
    if not run_dirs:
        raise ValueError("At least one experiment directory is required.")
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        root = Path(run_dir)
        run = load_experiment(root)
        metrics = aggregate_metrics(root)
        mi_df = run.get("mi_scores", pd.DataFrame())
        row = {
            "run": str(root),
            "experiment_id": root.name,
            "relations": metrics.get("relations", len(mi_df)),
            "mean_mi": metrics.get("mean_mi"),
            "best_mi": metrics.get("best_mi"),
            "best_relation": metrics.get("best_relation"),
            "mean_depthrank": metrics.get("mean_depthrank"),
            "best_threshold": metrics.get("best_threshold"),
            "best_n_shot": metrics.get("best_n_shot"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if "mean_mi" in df:
        df = df.sort_values("mean_mi", ascending=False, na_position="last").reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        baseline = (
            float(df.iloc[-1]["mean_mi"])
            if len(df) and pd.notna(df.iloc[-1]["mean_mi"])
            else 0.0
        )
        df["relative_improvement"] = df["mean_mi"].apply(
            lambda value: _relative_improvement(float(value), baseline) if pd.notna(value) else None
        )
    return df


def export_comparison(df: pd.DataFrame, output_dir: Path | str = ".") -> dict[str, Path]:
    """Export comparison output as CSV, LaTeX, and Markdown."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out / "comparison.csv",
        "latex": out / "comparison.tex",
        "markdown": out / "summary.md",
    }
    df.to_csv(paths["csv"], index=False)
    paths["latex"].write_text(df.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
    lines = ["# Experiment Comparison", "", _to_markdown(df), ""]
    if not df.empty:
        best = df.iloc[0]
        lines.extend([
            "## Summary",
            f"Best run: `{best['run']}` with mean MI {best['mean_mi']:.3f}.",
            "",
        ])
    paths["markdown"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def _relative_improvement(value: float, baseline: float) -> float | None:
    if abs(baseline) < 1e-12:
        return None
    return (value - baseline) / abs(baseline)


def _to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
