"""Experiment tracking adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow


@dataclass
class ExperimentTracker:
    """Thin MLflow-backed tracker used by experiment runners."""

    tracking_uri: str | None
    experiment_name: str

    def __enter__(self) -> "ExperimentTracker":
        """Start a tracking run."""

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """End the active tracking run."""

        mlflow.end_run(status="FAILED" if exc else "FINISHED")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log flat or nested parameters to MLflow."""

        def flatten(prefix: str, value: Any) -> dict[str, Any]:
            if isinstance(value, dict):
                return {k: v for key, item in value.items() for k, v in flatten(f"{prefix}{key}.", item).items()}
            return {prefix[:-1]: value}

        mlflow.log_params(flatten("", params))

    def log_artifact(self, path: Path | str) -> None:
        """Log a local artifact file."""

        mlflow.log_artifact(str(path))
