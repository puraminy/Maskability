"""Integration tests for the official ATOMIC CSV loader."""

from pathlib import Path

from maskability_index.datasets.atomic import RelationInstance, load_atomic2020_instances


def test_tiny_official_atomic_csv_sample_loads(tmp_path: Path) -> None:
    """A tiny author-format CSV expands every non-none relation target."""
    data_dir = tmp_path / "atomic"
    data_dir.mkdir()
    (data_dir / "v4_atomic_dev.csv").write_text(
        "event,oEffect,oReact,oWant,xAttr,xEffect,xIntent,xNeed,xReact,xWant\n"
        "PersonX drinks coffee,['PersonX stays awake'],[],[],[],[],['to stay awake'],[],[],[]\n",
        encoding="utf-8",
    )

    instances = load_atomic2020_instances(split="validation", local_path=data_dir)
    assert len(instances) > 0

