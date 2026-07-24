"""Reproduce manuscript experiments under results/reproduction/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from maskability_index.experiments import ExperimentRunner  # noqa: E402


def main() -> None:
    """Run every manuscript-reproduction experiment with the generic runner."""
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        overrides = ["experiment=reproduction", f"project.root={ROOT}"]
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(cfg)
    cfg.experiment.output_dir = "results/reproduction"
    ExperimentRunner(cfg, "results/reproduction").run()


if __name__ == "__main__":
    main()
