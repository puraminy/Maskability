"""Publication-ready report generation for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from maskability_index.evaluation.evaluator import aggregate_metrics, load_experiment
from maskability_index.plotting import generate_plots


def generate_report(
    run_dirs: list[Path | str], output_dir: Path | str = "results/report"
) -> dict[str, Path]:
    """Generate Markdown and LaTeX summaries plus copied/regenerated plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    all_mi: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        root = Path(run_dir)
        metrics = aggregate_metrics(root)
        run = load_experiment(root)
        rows.append({"run": str(root), **metrics})
        if "mi_scores" in run:
            df = run["mi_scores"].copy()
            df["run"] = str(root)
            all_mi.append(df)
    summary = pd.DataFrame(rows)
    combined = pd.concat(all_mi, ignore_index=True) if all_mi else pd.DataFrame()
    if not combined.empty:
        generate_plots(combined, out / "plots")
    best_model = _best(summary, "mean_mi", "run")
    best_mi = (
        float(summary["mean_mi"].max())
        if "mean_mi" in summary and not summary.empty
        else None
    )
    md = _markdown(summary, best_model, best_mi)
    tex = _latex(summary, best_model, best_mi)
    paths = {"markdown": out / "summary.md", "latex": out / "summary.tex"}
    paths["markdown"].write_text(md, encoding="utf-8")
    paths["latex"].write_text(tex, encoding="utf-8")
    return paths


def _best(df: pd.DataFrame, metric: str, label: str) -> str | None:
    if df.empty or metric not in df:
        return None
    row = df.sort_values(metric, ascending=False, na_position="last").iloc[0]
    return str(row[label])


def _markdown(summary: pd.DataFrame, best_model: str | None, best_mi: float | None) -> str:
    lines = ["# Maskability Analysis Report", ""]
    lines.append(f"Best model/run: `{best_model}`" if best_model else "Best model/run: unavailable")
    lines.append(f"Best MI: {best_mi:.3f}" if best_mi is not None else "Best MI: unavailable")
    for field, label in [("best_threshold", "Best threshold"), ("best_n_shot", "Best n-shot")]:
        value = (
            summary[field].dropna().iloc[0]
            if field in summary and summary[field].notna().any()
            else "unavailable"
        )
        lines.append(f"{label}: {value}")
    run_table = _to_markdown(summary) if not summary.empty else "No runs found."
    lines.extend(["", "## Runs", "", run_table, ""])
    return "\n".join(lines)


def _latex(summary: pd.DataFrame, best_model: str | None, best_mi: float | None) -> str:
    table = summary.to_latex(index=False, float_format="%.3f") if not summary.empty else ""
    return "\n".join([
        "% Automatically generated Maskability analysis report",
        f"% Best model/run: {best_model or 'unavailable'}",
        f"% Best MI: {best_mi:.3f}" if best_mi is not None else "% Best MI: unavailable",
        table,
    ])


def _to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
