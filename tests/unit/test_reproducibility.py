"""Tests for reproducibility metadata helpers."""

from maskability_index.utils.reproducibility import collect_environment_info, git_hash, set_seed


def test_seed_and_environment_info() -> None:
    """Seed setup and environment collection should complete successfully."""

    set_seed(123)
    info = collect_environment_info(".")
    assert "python" in info
    assert "git_hash" in info


def test_git_hash_returns_string() -> None:
    """Git hash helper should always return a string."""

    assert isinstance(git_hash("."), str)
