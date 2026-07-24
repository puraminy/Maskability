"""Tests for production ATOMIC2020 CSV dataset loading."""

from pathlib import Path

import pytest

from maskability_index.datasets.atomic import RelationInstance, load_atomic2020_instances


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
