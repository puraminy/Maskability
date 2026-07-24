"""Integration test for real ATOMIC2020 dataset availability."""

from pathlib import Path

import pytest

from maskability_index.datasets.atomic import load_atomic2020_instances


def test_real_atomic2020_dataset_loads_when_present() -> None:
    """Load the real converted dataset when available, otherwise skip."""
    local_path = Path("data/atomic2020")
    if not local_path.exists():
        pytest.skip("Real ATOMIC2020 data not present at data/atomic2020.")

    instances = load_atomic2020_instances(split="train", local_path=local_path, backend="local")

    assert instances
    assert all(
        instance.head and instance.relation and instance.tail for instance in instances[:100]
    )
