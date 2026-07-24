"""Tests for output directory helpers."""

from maskability_index.utils.io import ensure_output_tree


def test_ensure_output_tree(tmp_path) -> None:
    """Output helper should create root and subdirectories."""

    root = ensure_output_tree(tmp_path / "run", ["logs", "plots"])
    assert root.exists()
    assert (root / "logs").is_dir()
    assert (root / "plots").is_dir()
