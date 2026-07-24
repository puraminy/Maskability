"""Hydra experiment runner for infrastructure-only Milestone 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from maskability_index.tracking.tracker import ExperimentTracker
from maskability_index.utils.io import ensure_output_tree, write_config, write_json
from maskability_index.utils.logging import configure_logging
from maskability_index.utils.reproducibility import collect_environment_info, set_seed


def run_experiment(cfg: DictConfig) -> Path:
    """Run the configured experiment infrastructure without scientific computation."""

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ensure_output_tree(output_dir, list(cfg.experiment.outputs.create_subdirs))
    logger = configure_logging(output_dir / "logs" / "log.txt")
    set_seed(int(cfg.experiment.seed))
    write_config(output_dir / "config.yaml", cfg)
    env_info: dict[str, Any] = collect_environment_info(cfg.project.root)
    env_info["seed"] = int(cfg.experiment.seed)
    write_json(output_dir / "environment.json", env_info)
    write_json(output_dir / "metrics.json", {"status": "infrastructure_ready"})
    with ExperimentTracker(
        tracking_uri=cfg.experiment.tracking.mlflow_tracking_uri,
        experiment_name=cfg.experiment.tracking.experiment_name,
    ) as tracker:
        tracker.log_params(OmegaConf.to_container(cfg.experiment, resolve=True, throw_on_missing=True))
        tracker.log_artifact(output_dir / "config.yaml")
        tracker.log_artifact(output_dir / "environment.json")
    logger.info("Milestone 1 infrastructure run completed at %s", output_dir)
    return output_dir


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Command-line entry point for Hydra."""

    run_experiment(cfg)


if __name__ == "__main__":
    main()
