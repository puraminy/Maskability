"""Tests for production ATOMIC2020 dataset loading backends."""

from pathlib import Path

from maskability_index.datasets.atomic import RelationInstance, load_atomic2020_instances


def test_load_atomic2020_instances_prefers_local_converted_files(tmp_path: Path) -> None:
    """Local converted split files are normalized into RelationInstance rows."""
    data_dir = tmp_path / "atomic2020"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        '{"id":"r1","event":"PersonX cooks","xNeed":["buy food"],"xWant":["eat"]}\n',
        encoding="utf-8",
    )

    instances = load_atomic2020_instances(split="train", local_path=data_dir, backend="local")

    assert instances == [
        RelationInstance("PersonX cooks", "xNeed", "buy food", "train", "r1:xNeed:0"),
        RelationInstance("PersonX cooks", "xWant", "eat", "train", "r1:xWant:0"),
    ]
