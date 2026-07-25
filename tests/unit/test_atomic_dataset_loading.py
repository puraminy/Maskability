"""Tests for production ATOMIC2020 CSV dataset loading."""

from pathlib import Path

import pytest

from maskability_index.datasets.atomic import (
    RelationInstance,
    load_atomic2020_instances,
    sample_instances_per_relation,
)


def _write_atomic_csv(path: Path) -> None:
    path.write_text(
        "event,oEffect,oReact,oWant,xAttr,xEffect,xIntent,xNeed,xReact,xWant\n"
        "PersonX cooks,[],[],[],[],[],[],['buy food'],[],['eat']\n"
        "PersonX reads,[],[],[],[],[],[],['none'],[],[]\n",
        encoding="utf-8",
    )


def test_load_atomic2020_instances_reads_official_csv_files(tmp_path: Path) -> None:
    """Official ATOMIC split filenames are normalized into RelationInstance rows."""
    data_dir = tmp_path / "atomic"
    data_dir.mkdir()
    _write_atomic_csv(data_dir / "v4_atomic_trn.csv")

    instances = load_atomic2020_instances(split="train", local_path=data_dir, backend="csv")

    assert instances == [
        RelationInstance("PersonX cooks", "xNeed", "buy food", "train", "0:xNeed:0"),
        RelationInstance("PersonX cooks", "xWant", "eat", "train", "0:xWant:0"),
    ]


def test_load_atomic2020_instances_does_not_download_without_local_csv(tmp_path: Path) -> None:
    """The default backend fails locally instead of falling back to a download."""
    with pytest.raises(RuntimeError, match="automatic downloads are disabled"):
        load_atomic2020_instances(local_path=tmp_path / "missing")


def test_sample_instances_per_relation_keeps_each_relation() -> None:
    """Deterministic sampling applies the cap within each relation, not globally."""
    instances = [
        RelationInstance("h1", "xNeed", "t1", "validation", "1"),
        RelationInstance("h2", "xNeed", "t2", "validation", "2"),
        RelationInstance("h3", "xAttr", "t3", "validation", "3"),
        RelationInstance("h4", "xAttr", "t4", "validation", "4"),
    ]

    sampled = sample_instances_per_relation(instances, instances_per_relation=1)

    assert sampled == [instances[0], instances[2]]


def test_random_sample_instances_per_relation_is_seeded() -> None:
    """Random relation-balanced sampling is reproducible for reviewer sweeps."""
    instances = [
        RelationInstance(f"h{i}", relation, f"t{i}", "validation", str(i))
        for relation in ("xNeed", "xAttr")
        for i in range(6)
    ]

    first = sample_instances_per_relation(
        instances, instances_per_relation=2, strategy="random", seed=7
    )
    second = sample_instances_per_relation(
        instances, instances_per_relation=2, strategy="random", seed=7
    )
    different = sample_instances_per_relation(
        instances, instances_per_relation=2, strategy="random", seed=8
    )

    assert first == second
    assert first != different
    assert {instance.relation for instance in first} == {"xNeed", "xAttr"}
    assert len(first) == 4
