"""Integration tests for experiment comparison and report generation."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from maskability_index.evaluation import compare_runs, export_comparison, generate_report
from maskability_index.experiments import ExperimentRunner


def _run(tmp_path: Path, name: str) -> Path:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["experiment=reproduction", "project.root=."])
    OmegaConf.resolve(cfg)
    cfg.experiment.name = name
    cfg.experiment.output_dir = str(tmp_path / name)
    cfg.experiment.analysis.bootstrap_iterations = 10
    cfg.experiment.analysis.permutation_iterations = 10
    return ExperimentRunner(cfg).run()


def test_comparison_exports_csv_latex_markdown(tmp_path: Path) -> None:
    """Completed runs can be compared and exported in all required table formats."""
    first = _run(tmp_path, "E01")
    second = _run(tmp_path, "E02")
    df = compare_runs([first, second])
    outputs = export_comparison(df, tmp_path / "comparison")
    assert list(outputs) == ["csv", "latex", "markdown"]
    assert (tmp_path / "comparison" / "comparison.csv").exists()
    assert "\\begin{tabular}" in (tmp_path / "comparison" / "comparison.tex").read_text()
    assert "Experiment Comparison" in (tmp_path / "comparison" / "summary.md").read_text()


def test_report_generation_writes_summary_and_plots(tmp_path: Path) -> None:
    """Report generation writes summaries and publication figure formats."""
    run = _run(tmp_path, "E01")
    outputs = generate_report([run], tmp_path / "report")
    assert outputs["markdown"].exists()
    assert outputs["latex"].exists()
    assert (tmp_path / "report" / "plots" / "scatter.png").exists()
    assert (tmp_path / "report" / "plots" / "scatter.pdf").exists()
    assert (tmp_path / "report" / "plots" / "scatter.svg").exists()
