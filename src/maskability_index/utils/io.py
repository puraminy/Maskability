"""Filesystem helpers for experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def ensure_output_tree(output_dir: Path | str, subdirs: list[str]) -> Path:
    """Create an experiment output directory and requested child directories."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_config(path: Path | str, cfg: DictConfig) -> None:
    """Persist the resolved Hydra configuration as YAML."""

    Path(path).write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
