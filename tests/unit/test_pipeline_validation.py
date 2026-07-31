"""Scientific validation tests for the experiment runner."""

from __future__ import annotations

from omegaconf import OmegaConf
import pytest

from maskability_index.datasets.atomic import RelationInstance
from maskability_index.experiments.runner import ExperimentRunner


def _cfg():
    return OmegaConf.create(
        {
            "experiment": {
                "name": "validation-test",
                "seed": 1,
                "output_dir": "results/validation-test",
                "relations": {"mode": "selected", "selected": ["xWant", "xNeed"]},
                "dataset": {"train_split": "train", "evaluation_split": "test"},
                "few_shot": {
                    "enabled": True,
                    "n_samples": 2,
                    "strategy": "deterministic",
                    "seed": 1,
                },
                "prompting": {"style": "both", "n_shot": 2, "demonstrations": {"enabled": False}},
                "evaluation": {"depthrank": {"heads_per_relation": 1, "max_reference_tails": 1}},
                "model": {"name": "google-t5/t5-small"},
                "training": {"enabled": False},
                "analysis": {"threshold": 0.3},
                "outputs": {"generate_predictions": False},
            },
            "project": {"root": "."},
        }
    )


def _instance(head: str, relation: str, tail: str, split: str) -> RelationInstance:
    return RelationInstance(
        head=head,
        relation=relation,
        tail=tail,
        split=split,
        id=f"{split}:{head}:{relation}:{tail}",
    )


def test_train_heldout_overlap_raises_clear_error() -> None:
    runner = ExperimentRunner(_cfg())
    train = [_instance("same", "xWant", "tail", "train")]
    heldout = [_instance("same", "xWant", "tail", "test")]

    with pytest.raises(ValueError, match="overlap"):
        runner._validate_no_overlap(train, heldout, "train", "test")


def test_few_shot_training_set_requires_exact_n_per_relation() -> None:
    runner = ExperimentRunner(_cfg())
    train = [
        _instance("h1", "xWant", "t1", "train"),
        _instance("h2", "xWant", "t2", "train"),
        _instance("h3", "xNeed", "t3", "train"),
    ]

    with pytest.raises(ValueError, match="Few-shot invariant"):
        runner._construct_few_shot_training_set(train)


def test_few_shot_training_set_accepts_exact_balanced_relations() -> None:
    runner = ExperimentRunner(_cfg())
    train = [
        _instance("h1", "xWant", "t1", "train"),
        _instance("h2", "xWant", "t2", "train"),
        _instance("h3", "xNeed", "t3", "train"),
        _instance("h4", "xNeed", "t4", "train"),
    ]

    selected = runner._construct_few_shot_training_set(train)

    assert len(selected) == 4
    assert {instance.relation for instance in selected} == {"xWant", "xNeed"}
