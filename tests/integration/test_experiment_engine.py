"""Integration tests for the configuration-driven experiment engine."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from maskability_index.experiments import ExperimentRegistry, ExperimentRunner, default_registry


def _write_atomic_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "atomic"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "v4_atomic_dev.csv").write_text(
        "event,oEffect,oReact,oWant,xAttr,xEffect,xIntent,xNeed,xReact,xWant\n"
        "PersonX drinks coffee,[],[],[],[],[],[],[],[],['to stay awake']\n"
        "rain,['wet streets'],[],[],[],[],[],[],[],[]\n",
        encoding="utf-8",
    )
    return data_dir


def _cfg(tmp_path: Path):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["experiment=reproduction", "project.root=."])
    OmegaConf.resolve(cfg)
    cfg.experiment.output_dir = str(tmp_path / "reproduction")
    cfg.experiment.dataset.local_path = str(_write_atomic_fixture(tmp_path))
    cfg.experiment.dataset.backend = "local"
    cfg.experiment.analysis.bootstrap_iterations = 10
    cfg.experiment.analysis.permutation_iterations = 10
    return cfg


def test_reproduction_pipeline_creates_outputs(tmp_path: Path) -> None:
    """The reproduction pipeline creates every required artifact."""
    cfg = _cfg(tmp_path)
    out = ExperimentRunner(cfg).run()
    for relative in [
        "config.yaml",
        "metrics.json",
        "predictions.csv",
        "mi_scores.csv",
        "depthrank.csv",
        "plots/scatter.pdf",
        "plots/histogram.pdf",
        "plots/correlation.pdf",
        "plots/threshold.pdf",
        "plots/sensitivity.pdf",
        "latex/tables.tex",
        "latex/figures.tex",
        "logs/log.txt",
        "checkpoint/README.txt",
    ]:
        assert (out / relative).exists(), relative


def test_configuration_overrides_work(tmp_path: Path) -> None:
    """Hydra/OmegaConf overrides are reflected in persisted configs."""
    cfg = _cfg(tmp_path)
    cfg.experiment.analysis.threshold = 0.5
    out = ExperimentRunner(cfg).run()
    assert "threshold: 0.5" in (out / "config.yaml").read_text()


def test_registry_register_run_list(tmp_path: Path) -> None:
    """The experiment registry exposes register, run, and list."""
    cfg = _cfg(tmp_path)
    registry = default_registry(lambda c: ExperimentRunner(c).run())
    assert "E01" in registry.list()
    assert registry.run("E01", cfg).exists()
    custom = ExperimentRegistry()
    custom.register("E09", lambda c: Path(c.experiment.output_dir))
    assert custom.list() == ["E09"]
