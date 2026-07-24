"""Print and export deterministic demo prompts for ATOMIC2020 samples."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from maskability_index.datasets import RelationInstance, load_atomic2020_instances
from maskability_index.prompting import (
    FewShotPromptBuilder,
    MaskedPromptBuilder,
    PrefixPromptBuilder,
)


def main() -> None:
    """Load 10 random ATOMIC samples, print prompts, and write a CSV artifact."""
    rng = random.Random(13)
    instances = _load_demo_instances()
    samples = rng.sample(instances, k=min(10, len(instances)))

    prefix_builder = PrefixPromptBuilder()
    masked_builder = MaskedPromptBuilder()
    few_shot_builder = FewShotPromptBuilder(demonstrations=tuple(samples[:3]))

    rows: list[dict[str, str]] = []
    for sample in samples:
        row = {
            "relation": sample.relation,
            "head": sample.head,
            "tail": sample.tail,
            "prefix_prompt": prefix_builder.build(sample),
            "masked_prompt": masked_builder.build(sample),
            "few_shot_prompt": few_shot_builder.build(sample),
        }
        rows.append(row)
        print(f"relation: {row['relation']}")
        print(f"head: {row['head']}")
        print(f"tail: {row['tail']}")
        print(f"prefix prompt: {row['prefix_prompt']}")
        print(f"masked prompt: {row['masked_prompt']}")
        print(f"few-shot prompt: {row['few_shot_prompt']}")
        print("---")

    output = Path("results/demo_prompts.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_demo_instances() -> list[RelationInstance]:
    """Load ATOMIC2020 examples through the production dataset loader."""
    return load_atomic2020_instances(split="train", cache_dir="data/cache")


if __name__ == "__main__":
    main()
