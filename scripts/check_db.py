#!/usr/bin/env python3

import argparse
from pathlib import Path
from collections import Counter
from collections import defaultdict
from datasets import load_from_disk

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = SCRIPT_DIR.parent / "data" / "atomic2020_500"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", 
                        default=str(DEFAULT_INPUT))
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    local_path = Path(args.local_path).expanduser()
    ds = load_from_disk(str(local_path))

    # Get dataset name from folder
    dataset_name = local_path.name

    print(f"Dataset: {dataset_name}")
    print(f"Path: {local_path}")
    print()
    print(ds)
    print()

    # Collect all relations across all splits
    all_relations = []
    split_relations = {}

    for split_name, split in ds.items():
        if "relation" in split.column_names:
            relations = split["relation"]
            split_relations[split_name] = relations
            all_relations.extend(relations)

    # Count relations
    relation_counts = Counter(all_relations)

    print("=" * 80)
    print(f"Distinct Relations across all splits: {len(relation_counts)}")
    print("=" * 80)

    print("\nRelation distribution:")
    print("-" * 80)
    for relation, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {relation:25s}: {count:,}")

    print("\n" + "=" * 80)
    print("Per-split relation distribution:")
    print("=" * 80)

    for split_name, relations in split_relations.items():
        split_counts = Counter(relations)
        print(f"\n{split_name} ({len(relations):,} rows):")
        print("-" * 60)
        for relation, count in sorted(split_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {relation:25s}: {count:,}")
            
    heldout = ds["heldout"]
    stats = defaultdict(set)
    for row in heldout:
        stats[row["relation"]].add(row["head"])

    print("\nUnique heads in heldout")
    for rel in sorted(stats):
        print(f"{rel:20} {len(stats[rel])}")

    # Show example rows
    print("\n" + "=" * 80)
    print(f"First {args.rows} examples from each split:")
    print("=" * 80)

    for split_name, split in ds.items():
        print()
        print("=" * 80)
        print(f"Split: {split_name}")
        print("=" * 80)

        print("\nColumns:")
        print(split.column_names)

        print("\nFeatures:")
        print(split.features)

        print("\nNumber of rows:")
        print(len(split))

        print("\nFirst examples:\n")

        num_examples = min(args.rows, len(split))
        for i in range(num_examples):
            print(f"Example {i}")
            for k, v in split[i].items():
                print(f"  {k}: {repr(v)}")
            print()

        if num_examples < len(split):
            print(f"... and {len(split) - num_examples} more rows")


if __name__ == "__main__":
    main()
