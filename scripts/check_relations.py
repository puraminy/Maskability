#!/usr/bin/env python3
"""
CLI wrapper for relation-coverage diagnostics.

Works with Hugging Face DatasetDicts saved using Dataset.save_to_disk().

Example:
    python scripts/check_relations.py \
        --local-path ~/mask/data/atomic2020

    python scripts/check_relations.py \
        --local-path ~/mask/data/atomic2020 \
        --splits train validation

    python scripts/check_relations.py \
        --local-path ~/mask/data/atomic2020 \
        --json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk

from maskability_index.templates.registry import atomic2020_registry


def infer_relation_column(columns):
    """Infer the relation column name."""
    candidates = [
        "relation",
        "rel",
        "predicate",
    ]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"Could not find relation column. Available columns: {columns}"
    )


def infer_head_column(columns):
    """Infer the head/event column name."""
    candidates = [
        "head",
        "event",
        "source",
    ]
    for c in candidates:
        if c in columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local-path",
        required=True,
        help="Path to a DatasetDict saved with save_to_disk().",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to scan (default: all). Example: train validation",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON.",
    )

    args = parser.parse_args()

    dataset = load_from_disk(str(Path(args.local_path).expanduser()))

    available_splits = list(dataset.keys())

    splits = args.splits or available_splits

    for split in splits:
        if split not in dataset:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Available splits: {available_splits}"
            )

    registry = atomic2020_registry()
    registered_relations = set(registry.list_relations())

    counts = Counter()
    samples = defaultdict(list)

    total_instances = 0

    for split_name in splits:
        split = dataset[split_name]

        relation_col = infer_relation_column(split.column_names)
        head_col = infer_head_column(split.column_names)

        for row in split:
            relation = row[relation_col]

            counts[relation] += 1
            total_instances += 1

            if head_col is not None and len(samples[relation]) < 3:
                samples[relation].append(str(row[head_col]))

    detected_relations = set(counts.keys())

    unknown_relations = sorted(
        detected_relations - registered_relations
    )

    report = {
        "splits": splits,
        "total_instances": total_instances,
        "registered_relations": sorted(registered_relations),
        "counts": dict(counts),
        "samples": dict(samples),
        "unknown_relations": unknown_relations,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("Dataset relation coverage diagnostics")
    print("===================================")
    print(f"scanned splits: {splits}")
    print(f"total instances: {total_instances}")
    print()

    print(
        f"Registered relations ({len(registered_relations)}):"
    )
    for rel in sorted(registered_relations):
        print(f"  {rel}")

    print()

    print("Detected relations and counts:")
    print("------------------------------")

    for relation, count in counts.most_common():
        sample = ", ".join(samples.get(relation, []))
        print(
            f"{relation:20s} {count:8d}   sample heads: {sample!r}"
        )

    print()

    if unknown_relations:
        print("Unknown relations (no registered template):")
        for relation in unknown_relations:
            print(f"  {relation}")
    else:
        print(
            "All detected relations are covered by the template registry."
        )

    print()

    print(
        "If unknown relations are present, consider mapping them to "
        "canonical names or registering additional templates."
    )


if __name__ == "__main__":
    main()
