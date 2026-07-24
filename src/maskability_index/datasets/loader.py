"""Dataset loading wrappers built on HuggingFace Datasets."""

from __future__ import annotations

from typing import Any

try:
    from datasets import DatasetDict, load_dataset
except ImportError:  # pragma: no cover - exercised only without optional runtime deps
    DatasetDict = dict  # type: ignore[assignment]

    def load_dataset(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        raise ImportError("Install the `datasets` package to load HuggingFace datasets.")


def load_hf_dataset(path: str, cache_dir: str | None = None) -> DatasetDict:
    """Load a dataset by HuggingFace identifier without preprocessing assumptions."""

    dataset = load_dataset(path, cache_dir=cache_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict for {path!r}, got {type(dataset).__name__}.")
    return dataset
