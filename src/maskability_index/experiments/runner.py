"""Generic Hydra experiment runner for Maskability Index experiments."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from maskability_index.datasets.atomic import (
    RelationInstance,
    filter_instances_by_relations,
    load_atomic2020_instances,
    sample_heads_per_relation,
    sample_instances_per_relation,
)
from maskability_index.depthrank import DepthRankResult
from maskability_index.evaluation import update_results_index
from maskability_index.maskability import MaskabilityCalculator
from maskability_index.models.factory import create_seq2seq_model
from maskability_index.plotting import generate_plots
from maskability_index.prompting import (
    FewShotPromptBuilder,
    MaskedPromptBuilder,
    PrefixPromptBuilder,
)
from maskability_index.statistics import bootstrap_ci, correlations, permutation_test
from maskability_index.utils.io import ensure_output_tree, write_config, write_json
from maskability_index.utils.logging import configure_logging
from maskability_index.utils.reproducibility import collect_environment_info, set_seed

OUTPUT_SUBDIRS = ["plots", "latex", "logs", "checkpoint"]
PLOT_NAMES = [
    "scatter",
    "histogram",
    "correlation",
    "threshold",
    "sensitivity",
    "model_comparison",
    "baseline_comparison",
]
LOGGER = logging.getLogger(__name__)

TEMPLATE_FAMILIES = {"prompting": PrefixPromptBuilder, "masked_prompting": MaskedPromptBuilder}
STYLE_ALIASES = {"prefix": "prompting", "masked": "masked_prompting"}


class ExperimentRunner:
    """Configuration-driven orchestration for reproduction and reviewer experiments."""

    def __init__(self, cfg: DictConfig, output_dir: Path | str | None = None) -> None:
        """Store the resolved configuration and destination directory."""
        self.cfg = cfg
        name = str(cfg.experiment.name)
        root = Path(output_dir or cfg.experiment.get("output_dir", Path("results") / name))
        self.output_dir = root
        self._dataset_cache = None
        self._demo_cache = None

    def run(self) -> Path:
        """Execute dataset, prompting, inference, DepthRank, MI, stats, plots, and tables."""
        if self.cfg.experiment.get("sweep", {}).get("enabled", False):
            return self.run_sweeps()
        return self._run_single(self.cfg, self.output_dir)

    def run_sweeps(self) -> Path:
        """Execute configured sweep dimensions and write aggregate summaries."""
        ensure_output_tree(
            self.output_dir, ["sensitivity", "thresholds", "models", "prompts", "latex"]
        )
        write_config(self.output_dir / "config.yaml", self.cfg)
        sweep_rows: list[dict[str, Any]] = []
        for dimension, values, subdir in self._sweep_dimensions():
            for value in values:
                child_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
                label = self._sweep_label(dimension, value)
                self._apply_sweep_value(child_cfg, dimension, value)
                child_cfg.experiment.sweep.enabled = False
                child_cfg.experiment.output_dir = str(self.output_dir / subdir / label)
                out = ExperimentRunner(child_cfg, child_cfg.experiment.output_dir).run()
                mi_df = pd.read_csv(out / "mi_scores.csv")
                ci = bootstrap_ci(
                    mi_df["maskability_index"].tolist(),
                    seed=int(child_cfg.experiment.seed),
                    iterations=int(child_cfg.experiment.analysis.bootstrap_iterations),
                )
                sweep_rows.append(
                    {
                        "sweep": dimension,
                        "value": value,
                        "run_dir": str(out),
                        "relations": int(len(mi_df)),
                        "mean_MI": float(mi_df["maskability_index"].mean()),
                        "std_MI": float(mi_df["maskability_index"].std(ddof=0)),
                        "bootstrap_CI_lower": ci["lower"],
                        "bootstrap_CI_upper": ci["upper"],
                    }
                )
        summary = pd.DataFrame(sweep_rows)
        summary.to_csv(self.output_dir / "sweep_summary.csv", index=False)
        (self.output_dir / "latex" / "sweep_summary.tex").write_text(
            summary.to_latex(index=False, float_format="%.3f"), encoding="utf-8"
        )
        for dimension in sorted(summary["sweep"].unique()) if not summary.empty else []:
            dim_df = summary[summary["sweep"] == dimension]
            name = self._summary_name(dimension)
            dim_df.to_csv(self.output_dir / f"{name}.csv", index=False)
            (self.output_dir / "latex" / f"{name}.tex").write_text(
                dim_df.to_latex(index=False, float_format="%.3f"), encoding="utf-8"
            )
        return self.output_dir

    def _run_single(self, cfg: DictConfig, output_dir: Path) -> Path:
        """Execute one resolved experiment configuration."""
        start = time.time()
        ensure_output_tree(output_dir, OUTPUT_SUBDIRS)
        configure_logging(output_dir / "logs" / "log.txt")
        set_seed(int(cfg.experiment.seed))
        write_config(output_dir / "config.yaml", cfg)
        instances = self._load_dataset()
        assert len(instances) > 0, "No instances found!!"
        bundle = self._create_model_bundle()
        device = self._device()
        depthrank_calculator = self._create_depthrank_calculator(
            bundle.model, bundle.tokenizer, device
        )
        depthrank = self._compute_depthrank(instances, depthrank_calculator)
        if cfg.experiment.outputs.get("generate_predictions", False):
            predictions = self._build_predictions(
                depthrank,
                bundle.model,
                bundle.tokenizer,
                device,
            )
            self._write_csv(output_dir / "predictions.csv", predictions)

        mi_scores = self._compute_mi(depthrank)
        metrics = self._compute_statistics(mi_scores, start)
        self._write_csv(output_dir / "depthrank.csv", depthrank)
        mi_df = pd.DataFrame(mi_scores)
        mi_df.attrs["threshold"] = float(cfg.experiment.analysis.threshold)
        mi_df.to_csv(output_dir / "mi_scores.csv", index=False)
        write_json(output_dir / "metrics.json", metrics)
        self._write_summary_tables(mi_df, output_dir)
        generate_plots(mi_df, output_dir / "plots")
        self._write_latex(mi_df)
        update_results_index(output_dir, Path("results") / "index.json")
        (output_dir / "checkpoint" / "README.txt").write_text(
            "Checkpoint directory reserved for configured HuggingFace training outputs.\n",
            encoding="utf-8",
        )
        return output_dir

    def _load_dataset(self) -> list[RelationInstance]:
        if self._dataset_cache is not None:
            return self._dataset_cache
        dataset_cfg = self.cfg.experiment.dataset
        if dataset_cfg.get("backend", "auto") in {"atomic2020", "auto", "local", "hf", "csv"}:
            configured_backend = dataset_cfg.get("backend", "auto")
            backend = "auto" if configured_backend == "atomic2020" else configured_backend
            instances = load_atomic2020_instances(
                dataset_cfg.get("split", "validation"),
                dataset_cfg.cache_dir,
                dataset_cfg.hf_path,
                local_path=dataset_cfg.get("local_path", None),
                backend=backend,
            )
            filtered = self._filter_relations(instances)
            self._report_missing_relations(instances, filtered)
            sampled = self._sample_evaluation(filtered)
            self._dataset_cache = sampled
            return sampled
        return [RelationInstance(**dict(row)) for row in dataset_cfg.get("synthetic_rows", [])]

    def _filter_relations(self, instances: list[RelationInstance]) -> list[RelationInstance]:
        relations_cfg = self.cfg.experiment.get("relations", {})
        return filter_instances_by_relations(
            instances,
            mode=str(relations_cfg.get("mode", "selected")),
            selected=list(relations_cfg.get("selected", [])),
        )

    def _report_missing_relations(
        self, all_instances: list[RelationInstance], filtered: list[RelationInstance]
    ) -> None:
        relations_cfg = self.cfg.experiment.get("relations", {})
        if str(relations_cfg.get("mode", "selected")) != "selected":
            return
        available = {instance.relation for instance in all_instances}
        selected = list(relations_cfg.get("selected", []))
        missing = [relation for relation in selected if relation not in available]
        if missing:
            LOGGER.warning("Missing configured relations with no loaded examples: %s", missing)


    def _sample_dataset(self, instances: list[RelationInstance]) -> list[RelationInstance]:
        sampling = self.cfg.experiment.dataset.get("sampling", None)
        if sampling is None:
            legacy_limit = self.cfg.experiment.dataset.get("limit", None)
            return instances if legacy_limit is None else instances[: int(legacy_limit)]
        seed = sampling.get("seed", self.cfg.experiment.seed)
        return sample_instances_per_relation(
            instances,
            instances_per_relation=sampling.get("instances_per_relation", None),
            strategy=str(sampling.get("strategy", "deterministic")),
            seed=None if seed is None else int(seed),
        )

    def _sample_evaluation(self, instances: list[RelationInstance]) -> list[RelationInstance]:
        evaluation = self.cfg.experiment.get("evaluation", {})
        depthrank_cfg = evaluation.get("depthrank", {})
        heads = depthrank_cfg.get("heads_per_relation", None)
        max_tails = depthrank_cfg.get("max_reference_tails", None)
        seed = depthrank_cfg.get("seed", self.cfg.experiment.seed)
        few_shot_size = self._few_shot_size()
        if heads is not None and int(heads) == few_shot_size:
            LOGGER.warning(
                "DepthRank heads_per_relation (%s) equals few-shot size (%s); "
                "few-shot n is independent from evaluation size.",
                heads,
                few_shot_size,
            )
        if "depthrank" not in evaluation:
            limit = evaluation.get("max_instances_per_relation", "all")
            if str(limit).lower() != "all" and limit is not None:
                return sample_instances_per_relation(
                    instances,
                    instances_per_relation=int(limit),
                    strategy="deterministic",
                    seed=int(self.cfg.experiment.seed),
                )
        sampled = sample_heads_per_relation(
            instances,
            heads_per_relation=None if heads is None else int(heads),
            max_reference_tails=None if max_tails is None else int(max_tails),
            strategy=str(depthrank_cfg.get("strategy", "deterministic")),
            seed=None if seed is None else int(seed),
        )
        counts = pd.DataFrame([asdict(i) for i in sampled]) if sampled else pd.DataFrame()
        if not counts.empty:
            for relation, group in counts.groupby("relation"):
                LOGGER.info(
                    "Evaluating %s heads and %s reference tails for relation %s",
                    group["head"].nunique(),
                    len(group),
                    relation,
                )
        return sampled

    def _create_model_bundle(self):
        model_cfg = self.cfg.experiment.model
        return create_seq2seq_model(
            str(model_cfg.name),
            revision=str(model_cfg.get("revision", "main")),
            tokenizer_name=model_cfg.get("tokenizer_name", None),
        )

    def _create_depthrank_calculator(self, model: Any, tokenizer: Any, device: str):
        from maskability_index.depthrank import DepthRankCalculator

        return DepthRankCalculator(model, tokenizer, device=device)

    def _device(self) -> str:
        configured = self.cfg.experiment.model.get("device", None)
        if configured:
            return str(configured)
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _template_builders(self) -> dict[str, Any]:
        style = str(self.cfg.experiment.prompting.get("style", "both"))
        if style == "both":
            families = ("prompting", "masked_prompting")
        else:
            families = (STYLE_ALIASES.get(style, style),)
        demonstrations = self._demonstrations()
        builders: dict[str, Any] = {}
        for family in families:
            builder = TEMPLATE_FAMILIES[family]()
            if demonstrations:
                builder = FewShotPromptBuilder(
                    demonstrations=tuple(demonstrations),
                    base_builder=builder,
                )
            builders[family] = builder
        return builders

    def _demonstrations(self):
        if self._demo_cache is not None:
            return self._demo_cache

        if not self.cfg.experiment.few_shot.enabled:
            return []

        instances = self._load_dataset()

        demos = sample_instances_per_relation(
            instances,
            instances_per_relation=self._few_shot_size(),
            strategy="deterministic",
            seed=int(self.cfg.experiment.seed),
        )

        self._demo_cache = demos
        return demos

    def _few_shot_size(self) -> int:
        few_shot = self.cfg.experiment.get("few_shot", {})
        if few_shot:
            return int(few_shot.get("n_samples", 0))
        return int(self.cfg.experiment.prompting.get("n_shot", 0))

    def _compute_depthrank(
        self, instances: list[RelationInstance], calculator: Any
    ) -> list[dict[str, Any]]:
        """Compute DepthRank with the canonical teacher-forced calculator only."""
        rows: list[dict[str, Any]] = []
        for template_family, builder in self._template_builders().items():
            for instance in instances:
                prompt = builder.build(instance)
                result = calculator.compute(prompt, instance.tail)
                rows.append(self._depthrank_row(instance, template_family, result))
        return rows

    def _build_predictions(
        self, depthrank_rows: list[dict[str, Any]], model: Any, tokenizer: Any, device: str
    ) -> list[dict[str, Any]]:
        """Generate model predictions; never substitute the gold target as a prediction."""
        generation_length = int(self.cfg.experiment.training.get("generation_length", 32))
        beam_size = int(self.cfg.experiment.training.get("beam_size", 1))
        rows: list[dict[str, Any]] = []
        model.eval()
        for row in depthrank_rows:
            encoded = tokenizer(str(row["prompt"]), return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            try:
                import torch
            except ImportError:
                output_ids = model.generate(
                    **encoded,
                    max_length=generation_length,
                    num_beams=beam_size,
                )
            else:
                with torch.no_grad():
                    output_ids = model.generate(
                        **encoded,
                        max_length=generation_length,
                        num_beams=beam_size,
                    )
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            rows.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "relation": row["relation"],
                    "template_family": row["template_family"],
                    "style": row["style"],
                    "head": row["head"],
                    "tail": row["tail"],
                    "prompt": row["prompt"],
                    "target": row["target"],
                    "prediction": prediction,
                    "score": "",
                    "probability": "",
                }
            )
        return rows

    @staticmethod
    def _depthrank_row(
        instance: RelationInstance, template_family: str, result: DepthRankResult
    ) -> dict[str, Any]:
        style = "prefix" if template_family == "prompting" else "masked"
        return {
            "id": instance.id,
            "split": instance.split,
            "relation": instance.relation,
            "template_family": template_family,
            "style": style,
            "head": instance.head,
            "tail": instance.tail,
            "prompt": result.prompt,
            "target": result.target,
            "depthrank": result.depthrank,
            "target_tokens": list(result.tokens),
            "target_token_ids": list(result.token_ids),
            "token_ranks": list(result.token_ranks),
        }

    def _compute_mi(self, depthrank: list[dict[str, Any]]) -> list[dict[str, Any]]:
        calc = MaskabilityCalculator(threshold=float(self.cfg.experiment.analysis.threshold))
        by_relation: dict[str, dict[str, list[float]]] = {}
        for row in depthrank:
            relation_scores = by_relation.setdefault(
                str(row["relation"]), {"prompting": [], "masked_prompting": []}
            )
            family = str(
                row.get("template_family")
                or STYLE_ALIASES.get(str(row["style"]), str(row["style"]))
            )
            relation_scores[family].append(float(row["depthrank"]))
        rows: list[dict[str, Any]] = []
        for relation, values in sorted(by_relation.items()):
            sample_size = min(len(values["prompting"]), len(values["masked_prompting"]))
            result = asdict(
                calc.compute(
                    relation, values["prompting"], values["masked_prompting"], sample_size
                )
            )
            relation_rows = [row for row in depthrank if str(row["relation"]) == relation]
            heads = {str(row["head"]) for row in relation_rows}
            result.update(
                {
                    "evaluation_size": sample_size,
                    "few_shot_size": self._few_shot_size(),
                    "number_of_heads": len(heads),
                    "number_of_reference_tails": sample_size,
                    "model": str(self.cfg.experiment.model.name),
                    "seed": int(self.cfg.experiment.seed),
                }
            )
            rows.append(result)
        return rows

    def _compute_statistics(self, mi_scores: list[dict[str, Any]], start: float) -> dict[str, Any]:
        mi = [float(row["maskability_index"]) for row in mi_scores]
        drp = [float(row["dr_prompting"]) for row in mi_scores]
        drm = [float(row["dr_masked_prompting"]) for row in mi_scores]
        seed = int(self.cfg.experiment.seed)
        analysis = self.cfg.experiment.analysis
        stats_payload = {
            "environment": collect_environment_info(self.cfg.project.root),
            "seed": seed,
            "relations": len(mi_scores),
            "bootstrap_ci": bootstrap_ci(
                mi, seed=seed, iterations=int(analysis.bootstrap_iterations)
            ),
            "permutation_test": permutation_test(
                drp, drm, seed=seed, iterations=int(analysis.permutation_iterations)
            ),
            "runtime_seconds": time.time() - start,
        }
        stats_payload["correlations"] = {
            key: asdict(value) for key, value in correlations(drp, drm).items()
        }
        return stats_payload

    def _write_summary_tables(self, mi_df: pd.DataFrame, output_dir: Path) -> None:
        """Write sample-size summary tables for every single run."""
        summary = (
            mi_df.groupby("evaluation_size", as_index=False)["maskability_index"]
            .agg(mean_MI="mean", std_MI=lambda s: float(s.std(ddof=0)))
        )
        rows = []
        for sample_size in summary["evaluation_size"].tolist():
            values = mi_df.loc[
                mi_df["evaluation_size"] == sample_size, "maskability_index"
            ].tolist()
            ci = bootstrap_ci(
                values,
                seed=int(self.cfg.experiment.seed),
                iterations=int(self.cfg.experiment.analysis.bootstrap_iterations),
            )
            rows.append(
                {
                    "evaluation_size": sample_size,
                    "bootstrap_CI_lower": ci["lower"],
                    "bootstrap_CI_upper": ci["upper"],
                }
            )
        summary = summary.merge(pd.DataFrame(rows), on="evaluation_size")
        summary.to_csv(output_dir / "sample_size_summary.csv", index=False)
        (output_dir / "latex" / "sample_size_summary.tex").write_text(
            summary.to_latex(index=False, float_format="%.3f"), encoding="utf-8"
        )

    def _sweep_dimensions(self) -> list[tuple[str, list[Any], str]]:
        analysis = self.cfg.experiment.analysis
        configured = self.cfg.experiment.get("sweep", {}).get("dimensions", [])
        specs = {
            "instances_per_relation": (
                list(analysis.get("instances_per_relation", [])),
                "sensitivity",
            ),
            "threshold": (list(analysis.get("thresholds", [])), "thresholds"),
            "model": (list(analysis.get("models", [])), "models"),
            "prompt_variant": (list(analysis.get("prompt_variants", [])), "prompts"),
            "demonstrations": (list(analysis.get("n_shots", [])), "sensitivity"),
        }
        return [
            (dimension, values, subdir)
            for dimension, (values, subdir) in specs.items()
            if dimension in configured and values
        ]

    def _apply_sweep_value(self, cfg: DictConfig, dimension: str, value: Any) -> None:
        if dimension == "instances_per_relation":
            cfg.experiment.evaluation.depthrank.heads_per_relation = int(value)
        elif dimension == "threshold":
            cfg.experiment.analysis.threshold = float(value)
        elif dimension == "model":
            cfg.experiment.model.name = str(value)
            cfg.experiment.model.tokenizer_name = str(value)
        elif dimension == "prompt_variant":
            cfg.experiment.prompting.template_set = str(value)
        elif dimension == "demonstrations":
            cfg.experiment.prompting.demonstrations.enabled = int(value) > 0
            cfg.experiment.prompting.demonstrations.num_examples = int(value)

    @staticmethod
    def _sweep_label(dimension: str, value: Any) -> str:
        safe = str(value).replace("/", "_").replace(".", "_")
        prefix = "instances" if dimension == "instances_per_relation" else dimension
        return f"{prefix}_{safe}"

    @staticmethod
    def _summary_name(dimension: str) -> str:
        return {
            "instances_per_relation": "sample_size_summary",
            "threshold": "threshold_summary",
            "model": "model_summary",
            "prompt_variant": "prompt_summary",
            "demonstrations": "demonstration_summary",
        }[dimension]

    def _write_latex(self, mi_df: pd.DataFrame) -> None:
        latex = self.output_dir / "latex"
        table1 = mi_df.to_latex(
            index=False,
            float_format="%.3f",
            caption="Relation-level Maskability Index results.",
            label="tab:mi",
        )
        table2 = (
            mi_df.groupby("group", as_index=False)["maskability_index"]
            .agg(["count", "mean", "std"])
            .reset_index()
            .to_latex(
                index=False,
                float_format="%.3f",
                caption="Maskability groups summary.",
                label="tab:groups",
            )
        )
        (latex / "table_1.tex").write_text(table1, encoding="utf-8")
        (latex / "table_2.tex").write_text(table2, encoding="utf-8")
        (latex / "tables.tex").write_text(table1 + "\n" + table2, encoding="utf-8")
        (latex / "table_1.md").write_text(self._to_markdown(mi_df), encoding="utf-8")
        mi_df.to_csv(latex / "table_1.csv", index=False)
        figures = "\n".join(
            f"\\includegraphics[width=0.8\\linewidth]{{plots/{name}.pdf}}"
            for name in PLOT_NAMES
        )
        (latex / "figures.tex").write_text(figures + "\n", encoding="utf-8")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


    @staticmethod
    def _to_markdown(df: pd.DataFrame) -> str:
        headers = [str(col) for col in df.columns]
        rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines) + "\n"


def run_experiment(cfg: DictConfig) -> Path:
    """Hydra-compatible generic experiment entry point."""
    output_dir = None
    if hydra.core.hydra_config.HydraConfig.initialized():
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    return ExperimentRunner(cfg, output_dir).run()


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the configured experiment from the command line."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
