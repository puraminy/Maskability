"""Tests for the prompt demonstration script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from maskability_index.datasets import RelationInstance

MODULE_PATH = Path(__file__).resolve().parents[2] / "experiments" / "demo_prompts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("demo_prompts", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_prompt_loader_uses_production_dataset_loader(monkeypatch) -> None:
    """The demo should not maintain a bundled fallback dataset."""
    module = _load_module()
    expected = [RelationInstance("PersonX cooks", "xNeed", "buy food", "train", "1")]

    def production_loader(*args, **kwargs):
        assert kwargs == {"split": "train", "cache_dir": "data/cache"}
        return expected

    monkeypatch.setattr(module, "load_atomic2020_instances", production_loader)

    assert module._load_demo_instances() == expected
    assert not hasattr(module, "DEMO_FALLBACK_INSTANCES")
