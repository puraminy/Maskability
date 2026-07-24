"""Evaluation and analysis helpers for experiment outputs."""

from maskability_index.evaluation.comparison import compare_runs, export_comparison
from maskability_index.evaluation.evaluator import (
    aggregate_metrics,
    summarize_experiment,
    update_results_index,
)
from maskability_index.evaluation.report import generate_report

__all__ = [
    "aggregate_metrics",
    "compare_runs",
    "export_comparison",
    "generate_report",
    "summarize_experiment",
    "update_results_index",
]
