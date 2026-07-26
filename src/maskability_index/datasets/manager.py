"""Dataset manager utilities for Maskability datasets.

This module implements lightweight automatic HF download and local caching for
ATOMIC2020. It is intentionally small and uses the HuggingFace datasets API
when available; when not available the HF backend remains optional.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    from datasets import DatasetDict, load_dataset
except ImportError:  # pragma: no cover - optional dependency
    DatasetDict = dict  # type: ignore

    def load_dataset(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        raise ImportError("Install the `datasets` package to load HuggingFace datasets.")


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    name: str
    backend: str = "auto"  # auto | local | hf
    local_path: Optional[Path] = None
    hf_path: Optional[str] = None
    cache_dir: Optional[str] = None
    split: str = "train"


def hf_download_and_cache(hf_path: str, local_path: Path, cache_dir: Optional[str] = None) -> DatasetDict:
    """Download a HuggingFace dataset and save it to local_path using save_to_disk.

    If the dataset is already saved under local_path (from a previous run), load it
    from disk using load_from_disk instead of redownloading.
    """
    # If already saved to disk, load from disk
    if local_path.exists():
        # Rely on datasets' load_from_disk if the directory contains saved dataset
        try:
            from datasets import load_from_disk

            ds = load_from_disk(str(local_path))
            if not isinstance(ds, DatasetDict):
                raise TypeError("Expected DatasetDict when loading cached dataset from disk.")
            return ds
        except Exception:
            # Fall through to re-download if load_from_disk fails for any reason
            pass

    # Download from HF
    ds = load_dataset(hf_path, cache_dir=cache_dir)
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict for {hf_path!r}, got {type(ds).__name__}.")

    # Ensure parent dir exists
    local_path.mkdir(parents=True, exist_ok=True)
    try:
        ds.save_to_disk(str(local_path))
    except Exception:
        # If save_to_disk fails, ignore — we still return the dataset in memory
        pass
    return ds
