"""ATOMIC2020 dataset loading and normalization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
except ImportError:  # pragma: no cover - exercised only without optional runtime deps
    Dataset = Any  # type: ignore[misc, assignment]
    DatasetDict = dict  # type: ignore[assignment]

    def load_dataset(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        """Raise when HuggingFace Datasets is not installed."""
        raise ImportError("Install the `datasets` package to load ATOMIC2020.")

    def load_from_disk(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        """Raise when HuggingFace Datasets is not installed."""
        raise ImportError("Install the `datasets` package to load ATOMIC2020.")

ATOMIC2020_HF_PATH = "allenai/atomic2020"
DEFAULT_LOCAL_ATOMIC2020_PATH = "data/atomic2020"
SPLIT_ALIASES = {"dev": "validation", "valid": "validation", "val": "validation"}
_HEAD_COLUMNS = ("head", "event", "Event", "source", "input", "head_event")
_ID_COLUMNS = {"id", "ID"}
_LOCAL_SUFFIX_LOADERS = {".csv": "csv", ".json": "json", ".jsonl": "json"}
AtomicBackend = Literal["auto", "local", "hf"]


@dataclass(frozen=True, slots=True)
class RelationInstance:
    """A single ATOMIC relation triple with the original relation label preserved."""

    head: str
    relation: str
    tail: str
    split: str
    id: str


def canonical_split_name(split: str) -> str:
    """Map user-facing split aliases to HuggingFace split names."""
    return SPLIT_ALIASES.get(split, split)


def load_atomic2020_dataset(
    cache_dir: str | None = None,
    *,
    local_path: str | Path | None = None,
    hf_path: str = ATOMIC2020_HF_PATH,
    backend: AtomicBackend = "auto",
) -> DatasetDict:
    """Load ATOMIC2020, preferring local converted files and falling back to Hub data.

    Local data may be a ``datasets`` directory saved with ``save_to_disk`` or a directory
    containing split files such as ``train.csv``, ``validation.jsonl``, and ``test.parquet``.
    The HuggingFace path is used only when local data is unavailable or ``backend='hf'``.
    """
    errors: list[str] = []
    if backend not in {"auto", "local", "hf"}:
        raise ValueError("backend must be one of 'auto', 'local', or 'hf'.")

    if backend in {"auto", "local"}:
        candidate = Path(local_path or DEFAULT_LOCAL_ATOMIC2020_PATH)
        if candidate.exists():
            try:
                return _load_local_atomic2020_dataset(candidate, cache_dir=cache_dir)
            except Exception as exc:  # pragma: no cover - error message path
                if backend == "local":
                    raise
                errors.append(f"local {candidate}: {exc}")
        elif backend == "local":
            raise FileNotFoundError(f"Local ATOMIC2020 path does not exist: {candidate}")

    if backend in {"auto", "hf"}:
        try:
            dataset = load_dataset(hf_path, cache_dir=cache_dir)
            if not isinstance(dataset, DatasetDict):
                raise TypeError(
                    f"Expected DatasetDict for {hf_path!r}, got {type(dataset).__name__}."
                )
            return dataset
        except Exception as exc:
            if backend == "hf":
                raise
            errors.append(f"HuggingFace {hf_path}: {exc}")

    detail = " | ".join(errors) if errors else "no backend candidates were available"
    raise RuntimeError(
        "Could not load ATOMIC2020. Place converted files under "
        f"{DEFAULT_LOCAL_ATOMIC2020_PATH!r} or provide a loadable HuggingFace dataset. {detail}"
    )


def load_atomic2020_instances(
    split: str = "train",
    cache_dir: str | None = None,
    hf_path: str = ATOMIC2020_HF_PATH,
    *,
    local_path: str | Path | None = None,
    backend: AtomicBackend = "auto",
) -> list[RelationInstance]:
    """Load ATOMIC2020 as strongly typed relation instances."""
    dataset = load_atomic2020_dataset(
        cache_dir=cache_dir, local_path=local_path, hf_path=hf_path, backend=backend
    )
    hf_split = canonical_split_name(split)
    if hf_split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split {split!r} is unavailable. Available splits: {available}.")
    return list(iter_relation_instances(dataset[hf_split], split=split))


def iter_relation_instances(
    rows: Iterable[dict[str, Any]], split: str
) -> Iterable[RelationInstance]:
    """Normalize ATOMIC2020 rows from either long or wide dataset layouts."""
    for row_index, row in enumerate(rows):
        head = _first_text(row, _HEAD_COLUMNS)
        row_id = str(row.get("id", row.get("ID", row_index)))
        relation = row.get("relation") or row.get("rel")
        tail = row.get("tail") or row.get("target")
        if relation is not None and tail is not None:
            for tail_index, candidate in enumerate(_tails(tail)):
                if candidate:
                    suffix = "" if tail_index == 0 else f":{tail_index}"
                    yield RelationInstance(
                        _clean(head),
                        _clean(relation),
                        _clean(candidate),
                        split,
                        f"{row_id}{suffix}",
                    )
            continue
        for key, value in row.items():
            if key in {*_ID_COLUMNS, *_HEAD_COLUMNS}:
                continue
            for tail_index, candidate in enumerate(_tails(value)):
                if candidate:
                    yield RelationInstance(
                        _clean(head),
                        _clean(key),
                        _clean(candidate),
                        split,
                        f"{row_id}:{key}:{tail_index}",
                    )


def _load_local_atomic2020_dataset(path: Path, cache_dir: str | None = None) -> DatasetDict:
    if path.is_file():
        split = canonical_split_name(path.stem)
        return DatasetDict({split: _load_local_file(path, cache_dir=cache_dir)})

    if (path / "dataset_dict.json").exists():
        dataset = load_from_disk(str(path))
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected DatasetDict from {path}, got {type(dataset).__name__}.")
        return dataset

    data_files: dict[str, str] = {}
    parquet_files: dict[str, str] = {}
    for file_path in sorted(path.iterdir()):
        split = canonical_split_name(file_path.stem)
        if file_path.suffix in _LOCAL_SUFFIX_LOADERS:
            data_files[split] = str(file_path)
        elif file_path.suffix == ".parquet":
            parquet_files[split] = str(file_path)

    if data_files:
        extensions = {Path(file_name).suffix for file_name in data_files.values()}
        if len(extensions) != 1:
            raise ValueError(
                "Local CSV/JSON/JSONL split files must use one file type per directory."
            )
        dataset = load_dataset(
            _LOCAL_SUFFIX_LOADERS[extensions.pop()], data_files=data_files, cache_dir=cache_dir
        )
    elif parquet_files:
        dataset = load_dataset("parquet", data_files=parquet_files, cache_dir=cache_dir)
    else:
        raise FileNotFoundError(f"No ATOMIC2020 split files found in {path}.")

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict from {path}, got {type(dataset).__name__}.")
    return dataset


def _load_local_file(path: Path, cache_dir: str | None = None) -> Dataset:
    if path.suffix in _LOCAL_SUFFIX_LOADERS:
        dataset = load_dataset(
            _LOCAL_SUFFIX_LOADERS[path.suffix], data_files=str(path), cache_dir=cache_dir
        )
    elif path.suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(path), cache_dir=cache_dir)
    else:
        raise ValueError(f"Unsupported ATOMIC2020 local file type: {path.suffix}")
    return dataset["train"]


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key])
    raise ValueError(f"Could not identify ATOMIC head column in row keys: {sorted(row.keys())}.")


def _tails(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [] if value.strip().lower() in {"", "none", "[]"} else [value]
    if isinstance(value, dict):
        values: list[str] = []
        for nested in value.values():
            values.extend(_tails(nested))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_tails(item))
        return values
    return [str(value)]


def _clean(value: Any) -> str:
    return str(value).strip()
