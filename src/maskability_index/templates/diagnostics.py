"""Diagnostics for template/relationship coverage.

Provides a programmatic function to detect relation names present in a local
ATOMIC/ATOMIC2020 dataset that are not covered by the canonical template
registry. Also includes a small CLI script (in scripts/) to run the check and
print results.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

from maskability_index.datasets.atomic import RelationInstance, iter_relation_instances, load_atomic2020_dataset
from maskability_index.templates.registry import atomic2020_registry


def find_unknown_relations(
    local_path: str | Path,
    *,
    backend: str = "local",
    hf_path: str | None = None,
    cache_dir: str | None = None,
) -> Dict[str, Any]:
    """Scan the dataset at `local_path` and report relation coverage vs templates.

    Returns a dictionary with keys:
      - detected_relations: set of all relations found
      - registered_relations: set of relations the registry knows about
      - unknown_relations: detected - registered
      - counts: mapping relation -> occurrence count
      - samples: mapping relation -> up to 5 sample heads (strings)
      - splits: list of available split names
    """
    local_path = Path(local_path)
    ds = load_atomic2020_dataset(local_path=local_path, backend=backend, hf_path=hf_path, cache_dir=cache_dir)

    registry = atomic2020_registry()
    registered = set(registry.list_relations())

    counts: dict[str, int] = defaultdict(int)
    samples: dict[str, list[str]] = defaultdict(list)
    detected: set[str] = set()
    total_instances = 0
    splits = sorted(ds.keys())

    for split in splits:
        rows = ds[split]
        for inst in iter_relation_instances(rows, split=split):
            assert isinstance(inst, RelationInstance)
            rel = inst.relation
            detected.add(rel)
            counts[rel] += 1
            total_instances += 1
            if len(samples[rel]) < 5:
                samples[rel].append(inst.head)

    unknown = detected - registered

    return {
        "detected_relations": detected,
        "registered_relations": registered,
        "unknown_relations": unknown,
        "counts": dict(counts),
        "samples": dict(samples),
        "splits": splits,
        "total_instances": total_instances,
    }
