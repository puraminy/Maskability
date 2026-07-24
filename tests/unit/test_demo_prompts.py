"""Tests for the prompt demonstration script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "experiments" / "demo_prompts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("demo_prompts", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_prompt_loader_falls_back_when_hf_dataset_script_is_unsupported(monkeypatch) -> None:
    """The demo remains runnable when HuggingFace rejects script-backed ATOMIC repos."""
    module = _load_module()

    def unsupported_dataset(*args, **kwargs):
        raise RuntimeError("Dataset scripts are no longer supported, but found atomic.py")

    monkeypatch.setattr(module, "load_atomic2020_instances", unsupported_dataset)
    instances = module._load_demo_instances()
    assert len(instances) == len(module.DEMO_FALLBACK_INSTANCES)
    assert {instance.relation for instance in instances}
