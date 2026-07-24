"""Run the full Maskability Index pipeline on ATOMIC2020."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from maskability_index.datasets.atomic import RelationInstance, load_atomic2020_instances
from maskability_index.depthrank import DepthRankCalculator
from maskability_index.maskability import MaskabilityCalculator
from maskability_index.models.factory import create_seq2seq_model
from maskability_index.prompting.builders import MaskedPromptBuilder, PrefixPromptBuilder
from maskability_index.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Maskability Index experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="google/t5-base", help="HuggingFace model or checkpoint path."
    )
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer name or path.")
    parser.add_argument("--revision", default="main", help="Model revision.")
    parser.add_argument("--split", default="validation", help="ATOMIC2020 split to evaluate.")
    parser.add_argument("--cache-dir", default=None, help="Dataset/model cache directory.")
    parser.add_argument(
        "--output-dir", default="results/maskability", help="Directory for outputs."
    )
    parser.add_argument(
        "--max-per-relation", type=int, default=5, help="Sample size n per relation."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.30, help="Optional grouping threshold."
    )
    parser.add_argument("--seed", type=int, default=13, help="Deterministic seed.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda.")
    return parser.parse_args()


def select_instances(
    instances: list[RelationInstance], max_per_relation: int
) -> list[RelationInstance]:
    """Select the first n instances per relation in deterministic dataset order."""
    counts: dict[str, int] = defaultdict(int)
    selected: list[RelationInstance] = []
    for instance in instances:
        if counts[instance.relation] < max_per_relation:
            selected.append(instance)
            counts[instance.relation] += 1
    return selected


def main() -> None:
    """Load model, build prompts, compute DepthRank, compute MI, and store outputs."""
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = create_seq2seq_model(args.model, revision=args.revision, tokenizer_name=args.tokenizer)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    depthrank = DepthRankCalculator(bundle.model, bundle.tokenizer, device=device)
    mi_calculator = MaskabilityCalculator(threshold=args.threshold)

    instances = select_instances(
        load_atomic2020_instances(split=args.split, cache_dir=args.cache_dir), args.max_per_relation
    )
    builders = {"prompting": PrefixPromptBuilder(), "masked_prompting": MaskedPromptBuilder()}

    rows: list[dict[str, Any]] = []
    by_relation_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for template_family, builder in builders.items():
        for instance in instances:
            prompt = builder.build(instance)
            result = depthrank.compute(prompt, instance.tail)
            by_relation_family[instance.relation][template_family].append(result.depthrank)
            rows.append(
                {
                    "id": instance.id,
                    "split": instance.split,
                    "relation": instance.relation,
                    "template_family": template_family,
                    "head": instance.head,
                    "tail": instance.tail,
                    "prompt": prompt,
                    "depthrank": result.depthrank,
                    "target_tokens": list(result.tokens),
                    "target_token_ids": list(result.token_ids),
                    "token_ranks": list(result.token_ranks),
                }
            )

    mi_rows = []
    for relation, families in sorted(by_relation_family.items()):
        if "prompting" not in families or "masked_prompting" not in families:
            continue
        result = mi_calculator.compute(
            relation,
            families["prompting"],
            families["masked_prompting"],
            sample_size=min(len(families["prompting"]), len(families["masked_prompting"])),
        )
        mi_rows.append(asdict(result))

    pd.DataFrame(mi_rows).to_csv(output_dir / "mi_scores.csv", index=False)
    metrics = {
        "relations": {row["relation"]: row for row in mi_rows},
        "config": vars(args) | {"device": device},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(vars(args) | {"device": device}), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(output_dir / "depthrank_inputs.csv", index=False)


if __name__ == "__main__":
    main()
