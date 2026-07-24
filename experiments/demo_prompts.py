"""Print and export deterministic demo prompts for ATOMIC2020 samples."""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path

from maskability_index.datasets import RelationInstance, load_atomic2020_instances
from maskability_index.prompting import FewShotPromptBuilder, MaskedPromptBuilder, PrefixPromptBuilder


DEMO_FALLBACK_INSTANCES = [
    RelationInstance("PersonX bakes a cake", "xNeed", "buy ingredients", "demo", "demo-1"),
    RelationInstance("PersonX bakes a cake", "xIntent", "to make dessert", "demo", "demo-2"),
    RelationInstance("PersonX gives PersonY a gift", "xReact", "happy", "demo", "demo-3"),
    RelationInstance("PersonX gives PersonY a gift", "oReact", "grateful", "demo", "demo-4"),
    RelationInstance("umbrella", "ObjectUse", "staying dry", "demo", "demo-5"),
    RelationInstance("library", "AtLocation", "school", "demo", "demo-6"),
    RelationInstance("knife", "MadeOf", "metal", "demo", "demo-7"),
    RelationInstance("PersonX studies", "xWant", "pass the exam", "demo", "demo-8"),
    RelationInstance("rain", "Causes", "wet ground", "demo", "demo-9"),
    RelationInstance("PersonX apologizes", "xEffect", "is forgiven", "demo", "demo-10"),
]


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
    """Load ATOMIC2020 examples, falling back to bundled examples for offline demos.

    Recent `datasets` releases reject Hub repositories that still rely on dataset
    loading scripts. The public `allenai/atomic2020`/`allenai/atomic` repository can
    therefore fail before data rows are available. This demo should remain runnable
    as an onboarding smoke test, so it uses a small deterministic fixture when the
    remote dataset cannot be loaded.
    """

    try:
        return load_atomic2020_instances(split="train", cache_dir="data/cache")
    except Exception as exc:
        message = str(exc)
        known_demo_blocker = (
            "Dataset scripts are no longer supported" in message
            or "ProxyError" in type(exc).__name__
            or "ConnectionError" in type(exc).__name__
            or "ConnectError" in type(exc).__name__
        )
        if not known_demo_blocker:
            raise
        warnings.warn(
            "Could not load ATOMIC2020 from HuggingFace for this prompt demo; using "
            "bundled demo examples instead. Full experiments should use a local "
            "converted ATOMIC2020 dataset or a datasets version/environment that "
            "supports the source.",
            stacklevel=2,
        )
        return list(DEMO_FALLBACK_INSTANCES)


if __name__ == "__main__":
    main()
