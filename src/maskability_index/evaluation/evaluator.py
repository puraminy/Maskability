"""Experiment result loading, aggregation, and indexing utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from maskability_index.utils.reproducibility import git_hash


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """Normalized metadata and aggregate metrics for one completed experiment."""

    experiment_id: str
    timestamp: str
    git_commit: str
    model: str
    dataset: str
    configuration_hash: str
    metrics: dict[str, Any]
    paths: dict[str, str]


def load_experiment(run_dir: Path | str) -> dict[str, Any]:
    """Load the standard artifacts produced by an experiment run."""
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {root}")
    metrics_path = root / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Completed experiment is missing metrics.json: {root}")
    payload: dict[str, Any] = {
        "path": root,
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
    }
    for name in ["mi_scores", "depthrank", "predictions"]:
        csv_path = root / f"{name}.csv"
        if csv_path.exists():
            payload[name] = pd.read_csv(csv_path)
    return payload


def aggregate_metrics(run_dir: Path | str) -> dict[str, Any]:
    """Aggregate run-level metrics without introducing new scientific methods."""
    run = load_experiment(run_dir)
    mi_df = run.get("mi_scores", pd.DataFrame())
    depthrank_df = run.get("depthrank", pd.DataFrame())
    metrics = dict(run["metrics"])
    if not mi_df.empty:
        metrics.update(
            {
                "mean_mi": float(mi_df["maskability_index"].mean()),
                "best_mi": float(mi_df["maskability_index"].max()),
                "best_relation": str(
                    mi_df.sort_values("maskability_index", ascending=False).iloc[0]["relation"]
                ),
                "best_threshold": _best_threshold(mi_df),
                "best_n_shot": _best_by_column(mi_df, "sample_size"),
            }
        )
    if not depthrank_df.empty:
        metrics["mean_depthrank"] = float(depthrank_df["depthrank"].mean())
    return metrics


def summarize_experiment(
    run_dir: Path | str,
    *,
    config_hash: str = "unknown",
    repo_root: Path | str = ".",
) -> ExperimentSummary:
    """Create a stable index entry for a completed experiment directory."""
    root = Path(run_dir)
    metrics = aggregate_metrics(root)
    config = _read_config_metadata(root / "config.yaml")
    env = metrics.get("environment", {}) if isinstance(metrics.get("environment"), dict) else {}
    return ExperimentSummary(
        experiment_id=str(config.get("id") or root.name),
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=str(env.get("git_hash") or git_hash(repo_root)),
        model=str(config.get("model") or "unknown"),
        dataset=str(config.get("dataset") or "unknown"),
        configuration_hash=config_hash,
        metrics=metrics,
        paths={
            "root": str(root),
            "metrics": str(root / "metrics.json"),
            "mi_scores": str(root / "mi_scores.csv"),
            "depthrank": str(root / "depthrank.csv"),
            "plots": str(root / "plots"),
            "latex": str(root / "latex"),
        },
    )


def update_results_index(
    run_dir: Path | str, index_path: Path | str = "results/index.json"
) -> Path:
    """Add or replace a completed experiment entry in the global results index."""
    index = Path(index_path)
    index.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_experiment(run_dir, config_hash=_file_sha256(Path(run_dir) / "config.yaml"))
    entries: list[dict[str, Any]] = []
    if index.exists():
        existing = json.loads(index.read_text(encoding="utf-8"))
        entries = list(existing.get("experiments", []))
    key = str(summary.paths["root"])
    entries = [entry for entry in entries if entry.get("paths", {}).get("root") != key]
    entries.append(asdict(summary))
    payload = json.dumps({"experiments": entries}, indent=2, sort_keys=True) + "\n"
    index.write_text(payload, encoding="utf-8")
    return index


def _read_config_metadata(path: Path) -> dict[str, str]:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        exp = data.get("experiment", data)
        return {
            "id": exp.get("id"),
            "model": (exp.get("model") or {}).get("name"),
            "dataset": (exp.get("dataset") or {}).get("name"),
        }
    except Exception:
        return {}


def _file_sha256(path: Path) -> str:
    import hashlib

    if not path.exists():
        return "unknown"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _best_by_column(df: pd.DataFrame, column: str) -> Any:
    if column not in df:
        return None
    grouped = df.groupby(column)["maskability_index"].mean().sort_values(ascending=False)
    return grouped.index[0].item() if hasattr(grouped.index[0], "item") else grouped.index[0]


def _best_threshold(df: pd.DataFrame) -> float | None:
    if "threshold" in df:
        return _best_by_column(df, "threshold")
    return None
