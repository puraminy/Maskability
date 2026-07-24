"""ATOMIC2020 dataset loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    from datasets import DatasetDict, load_dataset
except ImportError:  # pragma: no cover - exercised only without optional runtime deps
    DatasetDict = dict  # type: ignore[assignment]

    def load_dataset(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        raise ImportError("Install the `datasets` package to load ATOMIC2020.")

ATOMIC2020_HF_PATH = "allenai/atomic2020"
SPLIT_ALIASES = {"dev": "validation", "valid": "validation", "val": "validation"}


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


def load_atomic2020_dataset(cache_dir: str | None = None) -> DatasetDict:
    """Download or read ATOMIC2020 through HuggingFace Datasets with local caching."""

    dataset = load_dataset(ATOMIC2020_HF_PATH, cache_dir=cache_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict for {ATOMIC2020_HF_PATH!r}, got {type(dataset).__name__}.")
    return dataset


def load_atomic2020_instances(
    split: str = "train", cache_dir: str | None = None, hf_path: str = ATOMIC2020_HF_PATH
) -> list[RelationInstance]:
    """Load ATOMIC2020 as strongly typed relation instances."""

    dataset = load_dataset(hf_path, cache_dir=cache_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict for {hf_path!r}, got {type(dataset).__name__}.")
    hf_split = canonical_split_name(split)
    if hf_split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split {split!r} is unavailable. Available splits: {available}.")
    return list(iter_relation_instances(dataset[hf_split], split=split))


def iter_relation_instances(rows: Iterable[dict[str, Any]], split: str) -> Iterable[RelationInstance]:
    """Normalize ATOMIC2020 rows from either long or wide dataset layouts."""

    for row_index, row in enumerate(rows):
        head = _first_text(row, ("head", "event", "Event", "source", "input", "head_event"))
        row_id = str(row.get("id", row.get("ID", row_index)))
        relation = row.get("relation") or row.get("rel")
        tail = row.get("tail") or row.get("target")
        if relation is not None and tail is not None:
            yield RelationInstance(_clean(head), _clean(relation), _clean(tail), split, row_id)
            continue
        for key, value in row.items():
            if key in {"id", "ID", "head", "event", "Event", "source", "input", "head_event"}:
                continue
            for tail_index, candidate in enumerate(_tails(value)):
                if candidate:
                    yield RelationInstance(
                        _clean(head), _clean(key), _clean(candidate), split, f"{row_id}:{key}:{tail_index}"
                    )


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
