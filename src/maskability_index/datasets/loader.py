"""Dataset loading wrappers built on HuggingFace Datasets."""

from __future__ import annotations

from datasets import DatasetDict, load_dataset


def load_hf_dataset(path: str, cache_dir: str | None = None) -> DatasetDict:
    """Load a dataset by HuggingFace identifier without preprocessing assumptions."""

    dataset = load_dataset(path, cache_dir=cache_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict for {path!r}, got {type(dataset).__name__}.")
    return dataset
