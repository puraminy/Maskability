"""Hydra/OmegaConf configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_config(config_dir: Path | str = "configs", config_name: str = "config") -> DictConfig:
    """Load an experiment configuration from YAML files with Hydra composition.

    Args:
        config_dir: Directory containing the Hydra config tree.
        config_name: Root configuration name without the `.yaml` suffix.

    Returns:
        A resolved `DictConfig` containing the full experiment configuration.
    """

    absolute_config_dir = Path(config_dir).resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(absolute_config_dir)):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """Convert a Hydra configuration to a plain Python dictionary."""

    return dict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
