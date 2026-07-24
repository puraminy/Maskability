"""Integration test for real ATOMIC2020 CSV dataset availability."""

from pathlib import Path

import pytest

from maskability_index.datasets.atomic import load_atomic2020_instances


def test_real_atomic2020_dataset_loads_when_present() -> None:
    """Load the real official CSV dataset when available, otherwise skip."""
    local_path = Path("data/atomic")
    if not (local_path / "v4_atomic_trn.csv").exists():
        pytest.skip("Real ATOMIC2020 CSV data not present at data/atomic.")
    instances = load_atomic2020_instances(split="train", local_path=local_path, backend="csv")
    assert instances
    assert all(instance.split == "train" for instance in instances[:100])
