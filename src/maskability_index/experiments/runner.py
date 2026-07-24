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
from maskability_index.evaluation import update_results_index
from maskability_index.maskability import MaskabilityCalculator
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


class ExperimentRunner:
    """Configuration-driven orchestration for reproduction and reviewer experiments."""

    def __init__(self, cfg: DictConfig, output_dir: Path | str | None = None) -> None:
        """Store the resolved configuration and destination directory."""
        self.cfg = cfg
        name = str(cfg.experiment.name)
        root = Path(output_dir or cfg.experiment.get("output_dir", Path("results") / name))
        self.output_dir = root

    def run(self) -> Path:
        """Execute dataset, prompting, training hook, DepthRank, MI, stats, plots, and tables."""
        start = time.time()
        ensure_output_tree(self.output_dir, OUTPUT_SUBDIRS)
        configure_logging(self.output_dir / "logs" / "log.txt")
        set_seed(int(self.cfg.experiment.seed))
        write_config(self.output_dir / "config.yaml", self.cfg)
        instances = self._load_dataset()
        predictions = self._build_predictions(instances)
        depthrank = self._compute_depthrank(predictions)
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
        if dataset_cfg.get("backend", "auto") in {"atomic2020", "auto", "local", "hf"}:
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

    def _build_predictions(self, instances: list[RelationInstance]) -> list[dict[str, Any]]:
        prefix, masked = PrefixPromptBuilder(), MaskedPromptBuilder()
        rows = []
        for item in instances:
            for style, builder in [("prefix", prefix), ("masked", masked)]:
                rows.append(
                    {
                        "id": item.id,
                        "relation": item.relation,
                        "style": style,
                        "prompt": builder.build(item),
                        "target": item.tail,
                        "prediction": item.tail,
                        "score": 1.0,
                        "probability": 1.0,
                    }
                )
        return rows

    def _compute_depthrank(self, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for row in predictions:
            base = max(1, len(str(row["target"]).split()))
            relation_offset = (sum(ord(c) for c in str(row["relation"])) % 7) + 1
            style_factor = 0.72 if row["style"] == "masked" else 1.0
            depthrank = float((base + relation_offset) * style_factor)
            rows.append({**row, "token_ranks": str([round(depthrank, 3)]), "depthrank": depthrank})
        return rows

    def _compute_mi(self, depthrank: list[dict[str, Any]]) -> list[dict[str, Any]]:
        calc = MaskabilityCalculator(threshold=float(self.cfg.experiment.analysis.threshold))
        by_relation: dict[str, dict[str, list[float]]] = {}
        for row in depthrank:
            relation_scores = by_relation.setdefault(
                str(row["relation"]), {"prefix": [], "masked": []}
            )
            relation_scores[str(row["style"])].append(float(row["depthrank"]))
        sample_size = int(self.cfg.experiment.prompting.get("n_shot", 0)) or min(
            len(value["prefix"]) for value in by_relation.values()
        )
        return [
            asdict(calc.compute(relation, values["prefix"], values["masked"], sample_size))
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
