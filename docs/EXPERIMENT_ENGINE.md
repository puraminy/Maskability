# Experiment Engine

The experiment engine provides one configuration-driven path for paper reproduction and reviewer experiments. Its design follows the project rule that datasets, prompts, models, thresholds, and output locations are declared in Hydra YAML rather than hardcoded in experiment scripts.

## Architecture

`maskability_index.experiments.ExperimentRunner` orchestrates the full pipeline:

1. load the Hydra configuration;
2. create the result directory tree;
3. initialize the configured dataset;
4. build prefix and masked prompts through the prompt builders;
5. reserve the configured checkpoint directory for fine-tuning artifacts;
6. compute or load DepthRank-compatible scores;
7. compute relation-level MI;
8. compute Pearson, Spearman, Kendall, bootstrap confidence intervals, and permutation tests;
9. generate PDF figures; and
10. export CSV, JSON, and LaTeX artifacts.

`ExperimentRegistry` maps reviewer/manuscript IDs (`E01`, `E02`, ...) to the generic runner. The default registry registers `E01` through `E08` without adding experiment-specific code paths.

## Adding experiments

Prefer adding a YAML file under `configs/experiment/`. A new reviewer experiment should only need code when it requires a new reusable pipeline stage rather than a new parameterization. At minimum, set:

```yaml
name: threshold_sensitivity
id: E03
seed: 13
output_dir: results/threshold_sensitivity
```

Then configure dataset, prompting, model, training, DepthRank, and analysis sections. Existing examples include:

- `reproduction.yaml`
- `threshold_sensitivity.yaml`
- `nshot_sensitivity.yaml`
- `model_comparison.yaml`
- `baseline_comparison.yaml`
- `prompt_robustness.yaml`
- `statistical_significance.yaml`

## Configuration

The active experiment is selected through Hydra:

```bash
python -m maskability_index.experiments.runner experiment=reproduction
```

The reproduction script is a thin wrapper around the same runner:

```bash
python experiments/reproduce_paper.py
```

By default the supplied configurations use a deterministic fixture backend so integration tests and dry runs do not download models or datasets. Set `experiment.dataset.backend=atomic2020` and enable the training/depthrank model stages when running full manuscript-scale experiments in a provisioned environment.

## Outputs

Every run creates:

```text
results/<experiment_name>/
  config.yaml
  metrics.json
  predictions.csv
  mi_scores.csv
  depthrank.csv
  plots/
    scatter.pdf
    histogram.pdf
    correlation.pdf
    threshold.pdf
    sensitivity.pdf
  latex/
    tables.tex
    figures.tex
  logs/
    log.txt
  checkpoint/
```

`metrics.json` includes reproducibility metadata (seed, Git hash, Python/platform/Torch details), correlation statistics, bootstrap confidence intervals, permutation tests, relation counts, and runtime.
