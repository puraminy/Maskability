"""Integration-test fixtures for lightweight experiment orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from maskability_index.depthrank import DepthRankResult


class _TensorLike(list):
    """Minimal object with a ``to`` method for runner generation code."""

    def to(self, device):
        return self


class _FakeTokenizer:
    """Small tokenizer implementing the methods used by generation."""

    def __call__(self, text=None, text_target=None, return_tensors=None, **kwargs):
        text = text_target if text_target is not None else text
        if isinstance(text, list):
            return {
                "input_ids": [[1, 2] for _ in text],
                "attention_mask": [[1, 1] for _ in text],
            }
        return {"input_ids": _TensorLike([[1, 2, 3]]), "attention_mask": _TensorLike([[1, 1, 1]])}

    def decode(self, ids, skip_special_tokens=True):
        return "generated:" + "-".join(str(int(token_id)) for token_id in ids)


class _FakeModel:
    """Deterministic seq2seq stub used only to avoid network/model downloads in tests."""

    def eval(self):
        return self

    def generate(self, **kwargs):
        return [[7, 8, 9]]


class _FakeDepthRankCalculator:
    """Deterministic test double for the canonical calculator interface."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def compute(self, prompt: str, target: str) -> DepthRankResult:
        masked_bonus = 1 if "<extra_id_0>" in prompt else 3
        return DepthRankResult(
            prompt=prompt,
            target=target,
            token_ids=(1,),
            tokens=("tok_1",),
            token_ranks=(masked_bonus,),
            depthrank=float(masked_bonus),
        )


class _FakeTrainer:
    """No-op trainer test double that records the intended training phase."""

    calls: list[str] = []

    def __init__(self, model, tokenizer, train_dataset, eval_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def train(self, resume_from_checkpoint=None):
        self.calls.append(f"train:{self.config.output_dir}")

    def evaluate(self):
        self.calls.append(f"evaluate:{self.config.output_dir}")
        return {"eval_loss": 0.0}

    def save_checkpoint(self, output_dir=None):
        from pathlib import Path

        path = Path(output_dir or self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "README.txt").write_text("fake checkpoint\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def fake_seq2seq_components(monkeypatch):
    """Patch heavyweight dependencies while preserving runner orchestration semantics."""

    monkeypatch.setattr(
        "maskability_index.experiments.runner.create_seq2seq_model",
        lambda *args, **kwargs: SimpleNamespace(model=_FakeModel(), tokenizer=_FakeTokenizer()),
    )
    monkeypatch.setattr(
        "maskability_index.experiments.runner.ExperimentRunner._create_depthrank_calculator",
        lambda self, model, tokenizer, device: _FakeDepthRankCalculator(model, tokenizer, device),
    )
    monkeypatch.setattr("maskability_index.experiments.runner.MaskabilitySeq2SeqTrainer", _FakeTrainer)
