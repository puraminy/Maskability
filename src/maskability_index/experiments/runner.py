"""Generic Hydra experiment runner for Maskability Index experiments."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig

from maskability_index.datasets.atomic import RelationInstance, load_atomic2020_instances
from maskability_index.depthrank import DepthRankResult
from maskability_index.evaluation import update_results_index
from maskability_index.maskability import MaskabilityCalculator
from maskability_index.models.factory import create_seq2seq_model
from maskability_index.plotting import generate_plots
from maskability_index.prompting import MaskedPromptBuilder, PrefixPromptBuilder
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

    def run(self) -> Path:
        """Execute dataset, prompting, inference, DepthRank, MI, stats, plots, and tables."""
        start = time.time()
        ensure_output_tree(self.output_dir, OUTPUT_SUBDIRS)
        configure_logging(self.output_dir / "logs" / "log.txt")
        set_seed(int(self.cfg.experiment.seed))
        write_config(self.output_dir / "config.yaml", self.cfg)
        instances = self._load_dataset()
        bundle = self._create_model_bundle()
        device = self._device()
        depthrank_calculator = self._create_depthrank_calculator(
            bundle.model, bundle.tokenizer, device
        )
        depthrank = self._compute_depthrank(instances, depthrank_calculator)
        predictions = self._build_predictions(depthrank, bundle.model, bundle.tokenizer, device)
        mi_scores = self._compute_mi(depthrank)
        metrics = self._compute_statistics(mi_scores, start)
        self._write_csv(self.output_dir / "predictions.csv", predictions)
        self._write_csv(self.output_dir / "depthrank.csv", depthrank)
        mi_df = pd.DataFrame(mi_scores)
        mi_df.attrs["threshold"] = float(self.cfg.experiment.analysis.threshold)
        mi_df.to_csv(self.output_dir / "mi_scores.csv", index=False)
        write_json(self.output_dir / "metrics.json", metrics)
        generate_plots(mi_df, self.output_dir / "plots")
        self._write_latex(mi_df)
        update_results_index(self.output_dir, Path("results") / "index.json")
        (self.output_dir / "checkpoint" / "README.txt").write_text(
            "Checkpoint directory reserved for configured HuggingFace training outputs.\n",
            encoding="utf-8",
        )
        return self.output_dir

    def _load_dataset(self) -> list[RelationInstance]:
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
            return instances[: int(dataset_cfg.get("limit", 20))]
        return [RelationInstance(**dict(row)) for row in dataset_cfg.get("synthetic_rows", [])]

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
        return {family: TEMPLATE_FAMILIES[family]() for family in families}

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
        sample_size = int(self.cfg.experiment.prompting.get("n_shot", 0)) or min(
            len(value["prompting"]) for value in by_relation.values()
        )
        return [
            asdict(
                calc.compute(
                    relation, values["prompting"], values["masked_prompting"], sample_size
                )
            )
            for relation, values in sorted(by_relation.items())
        ]

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
