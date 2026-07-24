"""Run DepthRank and Maskability Index evaluation on ATOMIC2020."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from maskability_index.datasets.atomic import RelationInstance, load_atomic2020_instances
from maskability_index.depthrank import DepthRankCalculator, maskability_index, mean_depthrank
from maskability_index.models.factory import create_seq2seq_model
from maskability_index.prompting.builders import MaskedPromptBuilder, PrefixPromptBuilder
from maskability_index.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the DepthRank experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="google/t5-base", help="HuggingFace model or checkpoint path."
    )
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer name or path.")
    parser.add_argument("--revision", default="main", help="Model revision.")
    parser.add_argument("--split", default="validation", help="ATOMIC2020 split to evaluate.")
    parser.add_argument("--cache-dir", default=None, help="Dataset/model cache directory.")
    parser.add_argument("--output-dir", default="results/depthrank", help="Directory for outputs.")
    parser.add_argument(
        "--max-per-relation", type=int, default=5, help="Sample size n per relation."
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
        if counts[instance.relation] >= max_per_relation:
            continue
        selected.append(instance)
        counts[instance.relation] += 1
    return selected


def main() -> None:
    """Load model/data, compute DepthRank for both template families, and save outputs."""
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = create_seq2seq_model(args.model, revision=args.revision, tokenizer_name=args.tokenizer)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    calculator = DepthRankCalculator(bundle.model, bundle.tokenizer, device=device)
    instances = select_instances(
        load_atomic2020_instances(split=args.split, cache_dir=args.cache_dir), args.max_per_relation
    )

    builders = {"prompting": PrefixPromptBuilder(), "masked_prompting": MaskedPromptBuilder()}
    rows: list[dict[str, Any]] = []
    for template_family, builder in builders.items():
        for instance in instances:
            prompt = builder.build(instance)
            result = calculator.compute(prompt, instance.tail)
            rows.append(
                {
                    "id": instance.id,
                    "split": instance.split,
                    "relation": instance.relation,
                    "template_family": template_family,
                    "head": instance.head,
                    "tail": instance.tail,
                    "prompt": prompt,
                    "target_tokens": list(result.tokens),
                    "target_token_ids": list(result.token_ids),
                    "token_ranks": list(result.token_ranks),
                    "depthrank": result.depthrank,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "depthrank.csv", index=False)

    metrics: dict[str, Any] = {"relations": {}}
    for relation, relation_frame in frame.groupby("relation"):
        by_family = {
            family: mean_depthrank(group["depthrank"].tolist())
            for family, group in relation_frame.groupby("template_family")
        }
        relation_metrics: dict[str, Any] = dict(by_family)
        if "prompting" in by_family and "masked_prompting" in by_family:
            relation_metrics["maskability_index"] = maskability_index(
                by_family["prompting"], by_family["masked_prompting"]
            )
        metrics["relations"][relation] = relation_metrics
    metrics["config"] = vars(args) | {"device": device}

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(vars(args) | {"device": device}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
