#!/usr/bin/env python3
"""
Convert a processed ATOMIC2020 DatasetDict into explicit triples with held-out split.

The dataset is split into:
- train: training examples (original train minus heldout, capped at max_per_relation)
- heldout: 100 unique heads per relation, capped at heldout-per-relation instances
- validation: original validation set
- test: original test set

Input:
    input  = "PersonX abuses PersonX's power oEffect [GEN]"
    output = "are told what to do"

Output:
    head      relation      tail

Examples
--------

Default (500 examples per relation in train, 100 examples per relation in heldout):

    python convert_db.py

Keep at most 1000 examples per relation in train:

    python convert_db.py --max-per-relation 1000

Custom heldout size:

    python convert_db.py \
        --heldout-per-relation 50

Specify paths:

    python convert_db.py \
        --input ~/mask/data/atomic2020 \
        --output ~/mask/data/atomic2020_500_heldout100
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

from pathlib import Path

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = SCRIPT_DIR.parent / "data" / "atomic2020"


RELATIONS = [
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
]

RELATIONS = sorted(RELATIONS, key=len, reverse=True)

PATTERN = re.compile(
    r"^(.*?)\s+("
    + "|".join(map(re.escape, RELATIONS))
    + r")\s+\[GEN\]$"
)


def create_heldout_split(train_split, heldout_per_relation, seed, max_heldout_instances=None):
    """
    Create heldout split by selecting unique heads per relation.
    
    For each relation:
    1. Group examples by head
    2. Shuffle unique heads deterministically
    3. Select first N heads for heldout
    4. Move ALL triples for those heads to heldout (or cap if specified)
    5. Keep remaining heads in train
    
    Args:
        train_split: Training split to split
        heldout_per_relation: Number of unique heads to holdout per relation
        seed: Random seed for reproducibility
        max_heldout_instances: Maximum instances per relation in heldout (default: None = unlimited)
    
    Returns:
        new_train: Dataset without heldout examples
        heldout: Dataset with heldout examples
        heldout_stats: dict mapping relation -> statistics
    """
    
    # Group examples by (relation, head)
    relation_head_groups = defaultdict(lambda: defaultdict(list))
    
    for idx, example in enumerate(train_split):
        inp = example.get("input")
        if inp is None:
            continue
            
        m = PATTERN.match(inp)
        if m is None:
            continue
            
        relation = m.group(2)
        head = m.group(1).strip()
        
        relation_head_groups[relation][head].append({
            "input": inp,
            "output": example.get("output", ""),
            "idx": idx
        })
    
    # For each relation, select heldout heads
    heldout_examples = []
    train_examples = []
    heldout_stats = {}
    
    import random
    random.seed(seed)
    
    for relation, head_groups in relation_head_groups.items():
        # Get unique heads
        unique_heads = list(head_groups.keys())
        
        # Shuffle deterministically
        random.shuffle(unique_heads)
        
        # Select heldout heads
        num_heldout_heads = min(heldout_per_relation, len(unique_heads))
        heldout_heads = set(unique_heads[:num_heldout_heads])
        
        # Collect all heldout examples for this relation
        relation_heldout_examples = []
        for head in heldout_heads:
            relation_heldout_examples.extend(head_groups[head])
        
        # Cap heldout instances if specified
        if max_heldout_instances is not None and len(relation_heldout_examples) > max_heldout_instances:
            # Shuffle the heldout examples (deterministically) and take first N
            random.shuffle(relation_heldout_examples)
            relation_heldout_examples = relation_heldout_examples[:max_heldout_instances]
        
        heldout_stats[relation] = {
            "total_heads": len(unique_heads),
            "heldout_heads": num_heldout_heads,
            "total_triples": sum(len(examples) for examples in head_groups.values()),
            "heldout_triples": len(relation_heldout_examples)
        }
        
        # Split examples
        # Remove heldout heads from train
        heldout_head_set = set(heldout_heads)
        for head, examples in head_groups.items():
            if head in heldout_head_set:
                # If we capped, we need to select which examples go to heldout
                # The ones in relation_heldout_examples are the selected ones
                if max_heldout_instances is not None:
                    # Only add to heldout if in the selected capped list
                    for ex in examples:
                        if ex in relation_heldout_examples:
                            heldout_examples.append(ex)
                        else:
                            train_examples.append(ex)
                else:
                    heldout_examples.extend(examples)
            else:
                train_examples.extend(examples)
    
    # Create new datasets
    def examples_to_dataset(examples):
        return Dataset.from_dict({
            "input": [ex["input"] for ex in examples],
            "output": [ex["output"] for ex in examples]
        })
    
    new_train = examples_to_dataset(train_examples)
    heldout = examples_to_dataset(heldout_examples)
    
    return new_train, heldout, heldout_stats


def convert_split(split, relation_counts, max_per_relation):
    """Convert a split to triple format."""
    
    heads = []
    relations = []
    tails = []
    
    failed_parse = 0
    missing_output = 0
    skipped_limit = 0
    
    for example in split:
        inp = example.get("input")
        out = example.get("output")
        
        if inp is None:
            failed_parse += 1
            continue
            
        m = PATTERN.match(inp)
        
        if m is None:
            failed_parse += 1
            continue
            
        relation = m.group(2)
        
        # Apply limit if specified
        if (
            max_per_relation is not None
            and relation_counts[relation] >= max_per_relation
        ):
            skipped_limit += 1
            continue
            
        relation_counts[relation] += 1
        
        if out is None:
            missing_output += 1
            out = ""
        
        heads.append(m.group(1).strip())
        relations.append(relation)
        tails.append(str(out).strip())
    
    print(
        f"    kept={len(heads):,} "
        f"failed={failed_parse:,} "
        f"missing_output={missing_output:,} "
        f"skipped_limit={skipped_limit:,}"
    )
    
    return Dataset.from_dict(
        {
            "head": heads,
            "relation": relations,
            "tail": tails,
        }
    )


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input DatasetDict (default: %(default)s)",
    )
    
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <input>_<max>_heldout<heldout>)",
    )
    
    parser.add_argument(
        "--max-per-relation",
        type=int,
        default=500,
        help="Maximum examples kept per relation in train (default: %(default)s)",
    )
    
    parser.add_argument(
        "--heldout-per-relation",
        type=int,
        default=100,
        help="Number of unique heads to holdout per relation (default: %(default)s)",
    )
    
    parser.add_argument(
        "--heldout-max-instances",
        type=int,
        default=None,
        help="Maximum instances per relation in heldout (default: None = unlimited)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic heldout split (default: %(default)s)",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input).expanduser()
    
    if args.output is None:
        suffix = (
            "all"
            if args.max_per_relation is None
            else str(args.max_per_relation)
        )
        heldout_suffix = f"_heldout{args.heldout_per_relation}"
        if args.heldout_max_instances is not None:
            heldout_suffix += f"_cap{args.heldout_max_instances}"
        output_path = input_path.parent / f"{input_path.name}_{suffix}{heldout_suffix}"
    else:
        output_path = Path(args.output).expanduser()
    
    print(f"Loading {input_path}")
    
    ds = load_from_disk(str(input_path))
    
    # Create heldout split from original train
    print(f"\nCreating heldout split ({args.heldout_per_relation} heads per relation, seed={args.seed})")
    if args.heldout_max_instances is not None:
        print(f"  Capping heldout instances to {args.heldout_max_instances} per relation")
    print("-" * 60)
    
    new_train, heldout, heldout_stats = create_heldout_split(
        ds["train"],
        args.heldout_per_relation,
        args.seed,
        args.heldout_max_instances
    )
    
    print("\nHeldout split statistics:")
    for relation, stats in sorted(heldout_stats.items()):
        print(f"  {relation:15s}: "
              f"{stats['heldout_triples']:,} triples from "
              f"{stats['heldout_heads']:,}/{stats['total_heads']:,} heads "
              f"(of {stats['total_triples']:,} total triples)")
    
    print(f"\nOriginal train: {len(ds['train']):,} → "
          f"Train: {len(new_train):,} + Heldout: {len(heldout):,}")
    
    # Build new DatasetDict
    new_ds = DatasetDict({
        "train": new_train,
        "heldout": heldout,
        "validation": ds["validation"],
        "test": ds["test"]
    })
    
    # Convert all splits to triple format
    print("\nConverting splits to triple format")
    print("-" * 60)
    
    converted = {}
    train_relation_counts = Counter()
    heldout_relation_counts = Counter()
    
    for split_name, split in new_ds.items():
        print(f"{split_name:10s} ({len(split):,} rows)")
        
        # Apply max_per_relation only to train split
        if split_name == "train":
            converted[split_name] = convert_split(
                split,
                train_relation_counts,
                args.max_per_relation
            )
        else:
            # No limit for heldout, validation, test (heldout was already capped during creation)
            if split_name == "heldout":
                # Track heldout relation counts for reporting
                converted[split_name] = convert_split(
                    split,
                    heldout_relation_counts,
                    None  # no limit - already capped
                )
            else:
                # Validation and test - use a dummy counter
                dummy_counter = Counter()
                converted[split_name] = convert_split(
                    split,
                    dummy_counter,
                    None  # no limit
                )
    
    out = DatasetDict(converted)
    
    out.save_to_disk(str(output_path))
    
    print(f"\nSaved to: {output_path}")
    
    print("\nTrain relations (capped at {} per relation)".format(args.max_per_relation))
    print("-" * 60)
    
    width = max(len(r) for r in train_relation_counts) if train_relation_counts else 20
    
    for relation in sorted(train_relation_counts):
        print(
            f"{relation:<{width}} : {train_relation_counts[relation]:>5}"
        )
    
    if heldout_relation_counts:
        print("\nHeldout relations (capped at {} per relation)".format(
            args.heldout_max_instances if args.heldout_max_instances is not None else "unlimited"
        ))
        print("-" * 60)
        for relation in sorted(heldout_relation_counts):
            print(
                f"{relation:<{width}} : {heldout_relation_counts[relation]:>5}"
            )
    
    # Also show heldout head statistics
    print("\nHeldout head statistics")
    print("-" * 60)
    for relation, stats in sorted(heldout_stats.items()):
        print(
            f"{relation:<{width}} : {stats['heldout_heads']:>5} heads "
            f"({stats['heldout_triples']:>5} triples kept out of {stats['total_triples']:,} total)"
        )


if __name__ == "__main__":
    main()
