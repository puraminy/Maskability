"""ATOMIC2020 CSV dataset loading and normalization."""

from __future__ import annotations

import ast
import csv
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

VALID_RELATIONS = {
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xWant",
    "oEffect",
    "oReact",
    "oWant",
}

ATOMIC2020_HF_PATH = "allenai/atomic2020"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_ATOMIC2020_PATH = PROJECT_ROOT / "data" / "atomic"

SPLIT_ALIASES = {"dev": "validation", "valid": "validation", "val": "validation"}
_OFFICIAL_SPLIT_FILES = {
    "train": "v4_atomic_trn.csv",
    "validation": "v4_atomic_dev.csv",
    "test": "v4_atomic_tst.csv",
}
_HEAD_COLUMNS = ("event", "head", "Event", "source", "input", "head_event")
_ID_COLUMNS = {"id", "ID"}
AtomicBackend = Literal["auto", "csv", "local", "hf"]


@dataclass(frozen=True, slots=True)
class RelationInstance:
    """A single ATOMIC relation triple with the original relation label preserved."""

    head: str
    relation: str
    tail: str
    split: str
    id: str


def canonical_split_name(split: str) -> str:
    """Map user-facing split aliases to canonical split names."""
    return SPLIT_ALIASES.get(split, split)


def sample_instances_per_relation(
    instances: Sequence[RelationInstance],
    *,
    instances_per_relation: int | None = None,
    strategy: str = "deterministic",
    seed: int | None = None,
) -> list[RelationInstance]:
    """Sample a configurable number of instances independently for each relation.

    ``deterministic`` preserves the loaded dataset order within each relation.
    ``random`` uses a local RNG seeded by ``seed`` so reviewer robustness sweeps are
    reproducible without mutating global random state. Relations with fewer than the
    requested number of examples keep all available examples.
    """
    if instances_per_relation is None:
        return list(instances)
    if instances_per_relation < 1:
        raise ValueError("instances_per_relation must be at least 1 when configured.")
    if strategy not in {"deterministic", "random"}:
        raise ValueError("sampling strategy must be 'deterministic' or 'random'.")

    by_relation: dict[str, list[RelationInstance]] = defaultdict(list)
    relation_order: list[str] = []
    for instance in instances:
        if instance.relation not in by_relation:
            relation_order.append(instance.relation)
        by_relation[instance.relation].append(instance)

    rng = random.Random(seed)
    sampled: list[RelationInstance] = []
    for relation in relation_order:
        relation_instances = list(by_relation[relation])
        if strategy == "random" and len(relation_instances) > instances_per_relation:
            selected = rng.sample(relation_instances, instances_per_relation)
            selected.sort(key=relation_instances.index)
        else:
            selected = relation_instances[:instances_per_relation]
        sampled.extend(selected)
    return sampled

def load_atomic2020_dataset(
    cache_dir: str | None = None,
    *,
    local_path: str | Path | None = None,
    hf_path: str = ATOMIC2020_HF_PATH,
    backend: AtomicBackend = "auto",
) -> Mapping[str, Any]:
    """Load ATOMIC2020 while preserving the historical dataset-returning API.

    The default backend reads the official ATOMIC CSV files from ``data/atomic`` and never
    downloads data. ``backend='hf'`` remains available as an explicit, separate future backend.
    """
    errors: list[str] = []
    if backend not in {"auto", "csv", "local", "hf"}:
        raise ValueError("backend must be one of 'auto', 'csv', 'local', or 'hf'.")

    if backend in {"auto", "csv", "local"}:
        candidate = Path(local_path or DEFAULT_LOCAL_ATOMIC2020_PATH)
        if candidate.exists():
            try:
                return _load_atomic_csv_dataset(candidate)
            except Exception as exc:
                if backend in {"csv", "local"}:
                    raise
                errors.append(f"CSV {candidate}: {exc}")
        elif backend in {"csv", "local"}:
            raise FileNotFoundError(f"Local ATOMIC CSV path does not exist: {candidate}")

    if backend == "hf":
        return _load_hf_atomic2020_dataset(hf_path, cache_dir=cache_dir)

    detail = " | ".join(errors) if errors else "no local CSV files were available"
    raise RuntimeError(
        "Could not load ATOMIC2020. Download and extract the official CSV files under "
        f"{DEFAULT_LOCAL_ATOMIC2020_PATH!r}; automatic downloads are disabled. {detail}"
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
    canonical_split = canonical_split_name(split)
    if canonical_split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split {split!r} is unavailable. Available splits: {available}.")
    return list(iter_relation_instances(dataset[canonical_split], split=split))


def iter_relation_instances(
    rows: Iterable[Mapping[str, Any]], split: str
) -> Iterable[RelationInstance]:
    """Normalize ATOMIC rows into one instance per ``(event, relation, target)`` pair."""
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
                        _clean(head), _clean(relation), candidate, split, f"{row_id}{suffix}"
                    )
            continue

        #for relation in VALID_RELATIONS:
        #    if relation not in row:
        #        continue

        #    value = row[relation]

        #    for tail_index, candidate in enumerate(_tails(value)):
        #        if candidate:
        #            yield RelationInstance(
        #                _clean(head),
        #                relation,
        #                candidate,
        #                split,
        #                f"{row_id}:{relation}:{tail_index}",
        #            )
        for key, value in row.items():
            if key in {*_ID_COLUMNS, *_HEAD_COLUMNS, "split", "prefix"}:
                continue
            for tail_index, candidate in enumerate(_tails(value)):
                if candidate:
                    yield RelationInstance(
                        _clean(head), _clean(key), candidate, split, f"{row_id}:{key}:{tail_index}"
                    )


def _load_atomic_csv_dataset(path: Path) -> dict[str, list[dict[str, Any]]]:
    if path.is_file():
        return {_split_from_file(path): _read_atomic_csv(path)}
    dataset: dict[str, list[dict[str, Any]]] = {}
    for split, filename in _OFFICIAL_SPLIT_FILES.items():
        file_path = path / filename
        if file_path.exists():
            dataset[split] = _read_atomic_csv(file_path)
    if not dataset:
        raise FileNotFoundError(
            f"No official ATOMIC CSV split files found in {path}. Expected: "
            + ", ".join(_OFFICIAL_SPLIT_FILES.values())
        )
    return dataset


def _load_hf_atomic2020_dataset(hf_path: str, cache_dir: str | None = None) -> Mapping[str, Any]:
    """Load an explicitly requested HuggingFace backend without using it by default."""
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:  # pragma: no cover - optional future backend only
        raise ImportError(
            "Install the `datasets` package to use the HuggingFace ATOMIC backend."
        ) from exc

    dataset = load_dataset(hf_path, cache_dir=cache_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict for {hf_path!r}, got {type(dataset).__name__}.")
    return dataset


def _split_from_file(path: Path) -> str:
    for split, filename in _OFFICIAL_SPLIT_FILES.items():
        if path.name == filename:
            return split
    return canonical_split_name(path.stem)


def _read_atomic_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key])
    raise ValueError(f"Could not identify ATOMIC head column in row keys: {sorted(row.keys())}.")


def _tails(value: Any) -> list[str]:
    """Parse official ATOMIC list-valued CSV cells and filter null targets."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "[]"}:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return [_clean(stripped)]
            return _tails(parsed)
        return [] if stripped.lower() == "none" else [_clean(stripped)]
    if isinstance(value, Mapping):
        values: list[str] = []
        for nested in value.values():
            values.extend(_tails(nested))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_tails(item))
        return values
    cleaned = _clean(value)
    return [] if cleaned.lower() == "none" else [cleaned]


def _clean(value: Any) -> str:
    return str(value).strip()
