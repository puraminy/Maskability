#!/usr/bin/env python3
"""
Convert a processed ATOMIC2020 DatasetDict into explicit triples.

Input:
    input  = "PersonX abuses PersonX's power oEffect [GEN]"
    output = "are told what to do"

Output:
    head      relation      tail

Optionally limit the number of examples kept for each relation.

Examples
--------

Default (500 examples per relation):

    python convert_db.py

Keep at most 1000 examples per relation:

    python convert_db.py --max-per-relation 1000

Specify paths:

    python convert_db.py \
        --input ~/mask/data/atomic2020 \
        --output ~/mask/data/atomic2020_1000
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

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


def convert_split(split, relation_counts, max_per_relation):

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
        help="Output directory (default: <input>_<N>)",
    )

    parser.add_argument(
        "--max-per-relation",
        type=int,
        default=500,
        help="Maximum examples kept per relation (default: %(default)s)",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()

    if args.output is None:

        suffix = (
            "all"
            if args.max_per_relation is None
            else str(args.max_per_relation)
        )

        output_path = input_path.parent / f"{input_path.name}_{suffix}"

    else:
        output_path = Path(args.output).expanduser()

    print(f"Loading {input_path}")

    ds = load_from_disk(str(input_path))

    relation_counts = Counter()

    converted = {}

    print()

    for split_name, split in ds.items():

        print(f"{split_name:10s} ({len(split):,} rows)")

        converted[split_name] = convert_split(
            split,
            relation_counts,
            args.max_per_relation,
        )

    out = DatasetDict(converted)

    out.save_to_disk(str(output_path))

    print("\nSaved to")
    print(output_path)

    print("\nExamples kept per relation\n")

    width = max(map(len, relation_counts))

    for relation in sorted(relation_counts):
        print(
            f"{relation:<{width}} : {relation_counts[relation]:>5}"
        )


if __name__ == "__main__":
    main()
