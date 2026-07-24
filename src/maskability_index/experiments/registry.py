"""Experiment registry for reviewer and reproduction experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

ExperimentCallable = Callable[[DictConfig], Path]


@dataclass(slots=True)
class ExperimentRegistry:
    """Deterministic registry keyed by manuscript/reviewer experiment IDs."""

    _experiments: dict[str, ExperimentCallable]

    def __init__(self) -> None:
        """Create an empty registry."""
        self._experiments = {}

    def register(self, experiment_id: str, runner: ExperimentCallable) -> None:
        """Register or replace an experiment implementation."""
        if not experiment_id.startswith("E") or not experiment_id[1:].isdigit():
            raise ValueError("Experiment IDs must use the E01, E02, ... format.")
        self._experiments[experiment_id] = runner

    def run(self, experiment_id: str, cfg: DictConfig) -> Path:
        """Run a registered experiment by ID."""
        if experiment_id not in self._experiments:
            available = ", ".join(self.list())
            raise KeyError(
                f"Unknown experiment {experiment_id!r}. Available: {available}."
            )
        return self._experiments[experiment_id](cfg)

    def list(self) -> list[str]:
        """List registered experiment IDs."""
        return sorted(self._experiments)


def default_registry(runner: ExperimentCallable) -> ExperimentRegistry:
    """Register all configured manuscript and reviewer experiment IDs to the generic runner."""
    registry = ExperimentRegistry()
    for index in range(1, 9):
        registry.register(f"E{index:02d}", runner)
    return registry
