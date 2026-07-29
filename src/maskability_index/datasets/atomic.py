"""ATOMIC and ATOMIC2020 CSV dataset loading and normalization with HF auto-download/cache.

Maintains backward compatibility with the original loader API while introducing:
- explicit dataset configurations (atomic vs atomic2020)
- backend=auto behavior: try local -> hf download -> save locally -> use local cached copy
- configurable HF path (default for atomic2020: Estwld/atomic2020-comet-origin)
"""
from __future__ import annotations

import ast
import csv
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Mapping as TypingMapping

from .manager import DatasetConfig, hf_download_and_cache

VALID_RELATIONS = {
    "AtLocation",
    "ObjectUse",
    "UsedFor",
    "CapableOf",
    "HasProperty",
    "FilledBy",
    "isFilledBy",
    "xAttr",
    "xIntent",
    "xNeed",
    "xWant",
    "xEffect",
    "xReact",
    "oEffect",
    "oReact",
    "oWant",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasSubEvent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "isAfter",
    "isBefore",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "PartOf",
    "ReceivesAction",
    "xReason",
}

import re

RELATION_PATTERN = re.compile(
    r"^(.*?)\s+("
    + "|".join(sorted(map(re.escape, VALID_RELATIONS), key=len, reverse=True))
    + r")\s+\[GEN\]$"
)

# Default HF dataset ids
ATOMIC_HF_PATH = "Estwld/atomic-comet-origin"  # placeholder; keep original atomic local by default
ATOMIC2020_HF_PATH = "Estwld/atomic2020-comet-origin"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_ATOMIC_PATH = PROJECT_ROOT / "data" / "atomic"
DEFAULT_LOCAL_ATOMIC2020_PATH = PROJECT_ROOT / "data" / "atomic2020_500"

SPLIT_ALIASES = {"dev": "validation", "valid": "validation", "val": "validation"}
_OFFICIAL_SPLIT_FILES = {
    "train": "v4_atomic_trn.csv",
    "validation": "v4_atomic_dev.csv",
    "test": "v4_atomic_tst.csv",
}
_HEAD_COLUMNS = ("event", "head", "Event", "source", "input", "head_event")
_ID_COLUMNS = {"id", "ID"}
AtomicBackend = Literal["auto", "csv", "arrow", "local", "hf"]


@dataclass(frozen=True, slots=True)
class RelationInstance:
    """A single ATOMIC relation triple with the original relation label preserved."""

    head: str
    relation: str
    tail: str
    split: str
    id: str


def canonical_split_name(split: str) -> str:
    return SPLIT_ALIASES.get(split, split)


def sample_instances_per_relation(
    instances: Sequence[RelationInstance],
    *,
    instances_per_relation: int | None = None,
    strategy: str = "deterministic",
    seed: int | None = None,
) -> list[RelationInstance]:
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


def sample_heads_per_relation(
    instances: Sequence[RelationInstance],
    *,
    heads_per_relation: int | None = None,
    max_reference_tails: int | None = None,
    strategy: str = "deterministic",
    seed: int | None = None,
) -> list[RelationInstance]:
    if heads_per_relation is not None and heads_per_relation < 1:
        raise ValueError("heads_per_relation must be at least 1 when configured.")
    if max_reference_tails is not None and max_reference_tails < 1:
        raise ValueError("max_reference_tails must be at least 1 when configured.")
    if strategy not in {"deterministic", "random"}:
        raise ValueError("sampling strategy must be 'deterministic' or 'random'.")

    by_relation_head: dict[str, dict[str, list[RelationInstance]]] = defaultdict(
        lambda: defaultdict(list)
    )
    relation_order: list[str] = []
    head_order: dict[str, list[str]] = defaultdict(list)
    for instance in instances:
        if instance.relation not in by_relation_head:
            relation_order.append(instance.relation)
        if instance.head not in by_relation_head[instance.relation]:
            head_order[instance.relation].append(instance.head)
        by_relation_head[instance.relation][instance.head].append(instance)

    rng = random.Random(seed)
    sampled: list[RelationInstance] = []
    for relation in relation_order:
        heads = list(head_order[relation])
        if (
            heads_per_relation is not None
            and strategy == "random"
            and len(heads) > heads_per_relation
        ):
            selected_heads = rng.sample(heads, heads_per_relation)
            selected_heads.sort(key=heads.index)
        else:
            selected_heads = heads[:heads_per_relation]
        for head in selected_heads:
            tails = by_relation_head[relation][head]
            sampled.extend(tails[:max_reference_tails])
    return sampled


def filter_instances_by_relations(
    instances: Sequence[RelationInstance], *, mode: str = "dataset", selected: Sequence[str] | None = None
) -> list[RelationInstance]:
    if mode not in {"all", "selected"}:
        raise ValueError("relations.mode must be one of 'all' or 'selected'.")
    if mode == "all":
        return list(instances)
    selected_set = set(selected or [])
    if not selected_set:
        raise ValueError("relations.selected must be non-empty when relations.mode='selected'.")
    return [instance for instance in instances if instance.relation in selected_set]


#
# Backwards compatible loader API for ATOMIC2020
#
def load_atomic2020_dataset(
    cache_dir: str | None = None,
    *,
    local_path: str | Path | None = None,
    hf_path: str | None = None,
    backend: AtomicBackend = "auto",
    split: str | None = None,
) -> TypingMapping[str, Any]:
    """Load ATOMIC2020 while preserving the historical dataset-returning API.

    New behavior (backend="auto"):
      1. Try local_path (default data/atomic2020)
      2. If not present or incomplete and backend="auto": download hf_path and save locally
      3. Return a mapping of split -> list[dict] (same shape as original CSV loader)
    """
    if backend not in {"auto", "csv", "arrow", "local", "hf"}:
        raise ValueError("backend must be one of 'auto', 'csv', 'arrow', 'local', or 'hf'.")

    hf_path = hf_path or ATOMIC2020_HF_PATH

    if local_path is None:
        candidate = DEFAULT_LOCAL_ATOMIC2020_PATH
    else:
        candidate = Path(local_path).expanduser()

        # If relative (e.g. data/atomic2020_500), make it project-root relative
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate

        # Optional: normalize absolute paths that are outside project root
        # leave them unchanged

    # Arrow/HF save_to_disk format
    print("Loading dataset...")
    dataset = None
    if not candidate.exists():
        print(f"WARNING: {candidate} doesn't exist")
    else:
        print(f"Info: local file: {candidate}")

    if backend in {"arrow", "local"}:
        if (candidate / Path(split) / "dataset_info.json").exists():
            dataset = _load_arrow_atomic2020_dataset(candidate, split)
        raise FileNotFoundError(f"Local ATOMIC2020 arrow path does not exist: {candidate}")

    # prefer local CSV files if backend explicitly csv/local
    elif backend in {"csv", "arrow", "local"}:
        if candidate.exists():
            dataset = _load_atomic_csv_dataset(candidate)
        raise FileNotFoundError(f"Local ATOMIC2020 CSV path does not exist: {candidate}")

    # backend == 'hf' -> direct HF load
    elif backend == "hf":
        dataset = _load_hf_atomic2020_dataset(hf_path, cache_dir=cache_dir)

    # backend == 'auto'
    elif backend == "auto":
        # 1. try Arrow dataset
        if candidate.exists() and (candidate / Path(split) / "dataset_info.json").exists():
            dataset = _load_arrow_atomic2020_dataset(candidate, split)

        # 2. try CSV
        elif candidate.exists():
            dataset = _load_atomic_csv_dataset(candidate)

        # 3. download from HF
        else:
            try:
                ds = hf_download_and_cache(hf_path, candidate, cache_dir=cache_dir)
                # ds is a DatasetDict (HF). Convert to list-of-dicts for compatibility
                dataset = {split: [dict(row) for row in ds[split]] for split in ds.keys()}
            except Exception as exc:
                # If HF attempt fails, present the same helpful message as before
                raise RuntimeError(
                    "Could not load ATOMIC2020. Attempted local CSV then HuggingFace download, "
                    f"but both failed. Last error: {exc}"
                ) from exc
    print(dataset.keys())
    print(dataset[split][0])
    return dataset

def load_atomic2020_instances(
    split: str = "train",
    cache_dir: str | None = None,
    hf_path: str | None = None,
    *,
    local_path: str | Path | None = None,
    backend: AtomicBackend = "auto",
) -> list[RelationInstance]:
    dataset = load_atomic2020_dataset(cache_dir=cache_dir, 
                                      local_path=local_path, 
                                      hf_path=hf_path, backend=backend, split=split)
    canonical_split = canonical_split_name(split)
    if canonical_split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split {split!r} is unavailable. Available splits: {available}.")
    return list(iter_relation_instances(dataset[canonical_split], split=split))

def iter_relation_instances(
    rows: Iterable[Mapping[str, Any]],
    split: str,
) -> Iterable[RelationInstance]:
    """
    Normalize ATOMIC datasets into RelationInstance objects.

    Supported formats
    -----------------

    1. Original ATOMIC
        event | xNeed | xWant | ...

    2. Explicit triples
        head | relation | tail

    3. Processed ATOMIC2020
        input | output
        input = "<head> <relation> [GEN]"
    """

    for row_index, row in enumerate(rows):

        row_id = str(row.get("id", row.get("ID", row_index)))

        #
        # ------------------------------------------------------------------
        # FORMAT 1 : Explicit tuples
        #
        # head | relation | tail
        # ------------------------------------------------------------------
        #

        relation = row.get("relation") or row.get("rel")
        tail = row.get("tail") or row.get("target")

        if relation is not None and tail is not None:

            head = _first_text(row, _HEAD_COLUMNS)

            for tail_index, candidate in enumerate(_tails(tail)):
                if candidate:
                    suffix = "" if tail_index == 0 else f":{tail_index}"

                    yield RelationInstance(
                        head=_clean(head),
                        relation=_clean(relation),
                        tail=candidate,
                        split=split,
                        id=f"{row_id}{suffix}",
                    )

            continue

        #
        # ------------------------------------------------------------------
        # FORMAT 2 : Processed ATOMIC2020
        #
        # input  = "... oEffect [GEN]"
        # output = "..."
        # ------------------------------------------------------------------
        #

        if "input" in row and "output" in row:

            match = RELATION_PATTERN.match(str(row["input"]))

            if match is not None:

                head = match.group(1).strip()
                relation = match.group(2)

                for tail_index, candidate in enumerate(_tails(row["output"])):

                    if candidate:

                        suffix = "" if tail_index == 0 else f":{tail_index}"

                        yield RelationInstance(
                            head=head,
                            relation=relation,
                            tail=candidate,
                            split=split,
                            id=f"{row_id}{suffix}",
                        )

                continue

        #
        # ------------------------------------------------------------------
        # FORMAT 3 : Original ATOMIC
        #
        # event | xNeed | xWant | ...
        # ------------------------------------------------------------------
        #

        head = _first_text(row, _HEAD_COLUMNS)

        for key, value in row.items():

            if key in {*_ID_COLUMNS, *_HEAD_COLUMNS, "split", "prefix"}:
                continue

            for tail_index, candidate in enumerate(_tails(value)):

                if not candidate:
                    continue

                yield RelationInstance(
                    head=_clean(head),
                    relation=_clean(key),
                    tail=candidate,
                    split=split,
                    id=f"{row_id}:{key}:{tail_index}",
                )

def _load_arrow_atomic2020_dataset(path: Path, split:str) -> TypingMapping[str, Any]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise ImportError(
            "Install the `datasets` package to use Arrow datasets."
        ) from exc

    print("Loading arrow dataset from disk...")
    ds_path = path / Path(split)
    dataset = load_from_disk(str(ds_path))

    if hasattr(dataset, "keys"):  # DatasetDict
        return {
            split: [dict(row) for row in dataset[split]]
            for split in dataset.keys()
        }

    # Single Dataset
    return {
        canonical_split_name(split): [dict(row) for row in dataset]
    }

def _load_atomic_csv_dataset(path: Path) -> dict[str, list[dict[str, Any]]]:
    if path.is_file():
        return {_split_from_file(path): _read_atomic_csv(path)}
    dataset: dict[str, list[dict[str, Any]]] = {}
    # Accept both official split filenames and generic CSVs in the directory
    for split, filename in _OFFICIAL_SPLIT_FILES.items():
        file_path = path / filename
        if file_path.exists():
            dataset[split] = _read_atomic_csv(file_path)
    # fallback: any csv file in directory -> infer split from name
    if not dataset:
        for child in sorted(path.iterdir()):
            if child.suffix.lower() == ".csv":
                dataset[_split_from_file(child)] = _read_atomic_csv(child)
    if not dataset:
        raise FileNotFoundError(
            f"No ATOMIC CSV split files found in {path}. Expected: " + ", ".join(_OFFICIAL_SPLIT_FILES.values())
        )
    return dataset


def _load_hf_atomic2020_dataset(hf_path: str, cache_dir: str | None = None) -> TypingMapping[str, Any]:
    """Load from HF without caching locally (explicit hf backend)."""
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise ImportError("Install the `datasets` package to use the HuggingFace ATOMIC backend.") from exc

    dataset = load_dataset(hf_path, cache_dir=cache_dir)
    if not isinstance(dataset, dict):  # DatasetDict under the hood behaves like dict
        raise TypeError(f"Expected DatasetDict for {hf_path!r}, got {type(dataset).__name__}.")
    # convert to mapping of split->list[dict]
    return {split: [dict(row) for row in dataset[split]] for split in dataset.keys()}


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
