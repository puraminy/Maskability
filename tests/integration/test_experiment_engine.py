"""Integration tests for the configuration-driven experiment engine."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from maskability_index.experiments import ExperimentRegistry, ExperimentRunner, default_registry


def _write_atomic_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "atomic"
    data_dir.mkdir(exist_ok=True)
    csv_text = (
        "event,oEffect,oReact,oWant,xAttr,xEffect,xIntent,xNeed,xReact,xWant\n"
        "PersonX drinks coffee,[],[],[],[],[],[],[],[],['to stay awake']\n"
        "rain,['wet streets'],[],[],[],[],[],[],[],[]\n"
    )
    (data_dir / "v4_atomic_trn.csv").write_text(csv_text, encoding="utf-8")
    (data_dir / "v4_atomic_dev.csv").write_text(csv_text, encoding="utf-8")
    (data_dir / "v4_atomic_tst.csv").write_text(csv_text, encoding="utf-8")
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


def test_relation_filtering_and_evaluation_limit_are_independent(tmp_path: Path) -> None:
    """Selected relations and evaluation caps affect MI sample size without prompt changes."""
    cfg = _cfg(tmp_path)
    cfg.experiment.relations.mode = "selected"
    cfg.experiment.relations.selected = ["xWant"]
    cfg.experiment.dataset.sampling.instances_per_relation = None
    cfg.experiment.evaluation.max_instances_per_relation = 1

    out = ExperimentRunner(cfg).run()
    mi = (out / "mi_scores.csv").read_text()

    assert "xWant" in mi
    assert "rain" not in mi
    assert ",1," in mi


def test_sweep_engine_creates_complete_child_outputs_and_summary(tmp_path: Path) -> None:
    """The generic sweep engine writes complete child runs and aggregate tables."""
    cfg = _cfg(tmp_path)
    cfg.experiment.output_dir = str(tmp_path / "sweep")
    cfg.experiment.sweep.enabled = True
    cfg.experiment.sweep.dimensions = ["instances_per_relation", "threshold"]
    cfg.experiment.analysis.instances_per_relation = [1]
    cfg.experiment.analysis.thresholds = [0.2, 0.4]

    out = ExperimentRunner(cfg).run()

    assert (out / "sensitivity" / "instances_1" / "mi_scores.csv").exists()
    assert (out / "thresholds" / "threshold_0_2" / "mi_scores.csv").exists()
    assert (out / "threshold_summary.csv").exists()
    assert (out / "sample_size_summary.csv").exists()


def test_registry_register_run_list(tmp_path: Path) -> None:
    """The experiment registry exposes register, run, and list."""
    cfg = _cfg(tmp_path)
    registry = default_registry(lambda c: ExperimentRunner(c).run())
    assert "E01" in registry.list()
    assert registry.run("E01", cfg).exists()
    custom = ExperimentRegistry()
    custom.register("E09", lambda c: Path(c.experiment.output_dir))
    assert custom.list() == ["E09"]
