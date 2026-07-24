"""Tests for Hydra configuration loading."""

from pathlib import Path

from maskability_index.config.loader import load_config


def test_load_config_resolves_defaults() -> None:
    """The default config should resolve the Milestone 1 experiment."""

    cfg = load_config(Path("configs"))
    assert cfg.experiment.name == "milestone1_infrastructure"
    assert cfg.experiment.model.name == "google/t5-small"
