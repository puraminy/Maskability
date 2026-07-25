# Experiment Engine

The experiment engine provides one configuration-driven path for paper reproduction and reviewer experiments. Its design follows the project rule that datasets, prompts, models, thresholds, and output locations are declared in Hydra YAML rather than hardcoded in experiment scripts.

## Architecture

`maskability_index.experiments.ExperimentRunner` orchestrates the full pipeline:

1. load the Hydra configuration;
2. create the result directory tree;
3. initialize the configured dataset and apply relation-balanced sampling;
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

Then configure dataset, prompting, model, training, DepthRank, and analysis sections. Dataset sampling is configured separately from prompt construction; `dataset.sampling.instances_per_relation` controls evaluation sample size, while `prompting.n_shot` is reserved for zero-shot/few-shot demonstration count. Existing examples include:

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

By default the reproduction configuration uses `google-t5/t5-base`, evaluates a deterministic number of examples per relation, and routes DepthRank through the teacher-forced seq2seq model implementation used by `experiments/run_maskability.py`. Integration tests patch the model factory with a local test double, but production experiments must load a configured model through `create_seq2seq_model()`; no deterministic fixture or synthetic DepthRank backend is used by the runner.

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

## Configurable research framework extensions

Current experiments can make the paper reproduction one configuration among many by using explicit Hydra sections:

```yaml
relations:
  mode: dataset      # dataset | all | selected
  selected: []

dataset:
  sampling:
    strategy: deterministic  # deterministic | random
    instances_per_relation: 5
    seed: ${experiment.seed}

evaluation:
  max_instances_per_relation: all  # integer or all

prompting:
  demonstrations:
    enabled: false
    num_examples: 0
    strategy: deterministic
    seed: ${experiment.seed}

sweep:
  enabled: true
  dimensions: [instances_per_relation, threshold, model, prompt_variant, demonstrations]
```

Relation selection is applied after loading and before sampling. `all` evaluates every relation present in the loaded split, while `selected` evaluates only the explicitly listed relation names. Few-shot sampling and held-out DepthRank evaluation sampling have independent seeds. `evaluation_size` in `mi_scores.csv` is computed from the actual number of prefix/masked DepthRank rows that contributed to each relation-level MI value; `few_shot_size` reports the independent few-shot condition.

When `sweep.enabled` is true, the runner executes one complete child experiment per selected sweep value and writes aggregate summaries such as `sample_size_summary.csv`, `threshold_summary.csv`, `model_summary.csv`, and matching LaTeX tables at the sweep root. Each child run still contains the full standard output set: config, metrics, predictions, DepthRank rows, MI scores, plots, and LaTeX tables.

## Few-shot size is not the DepthRank evaluation size

The experiment configuration intentionally separates the paper's two uses of sample
count. `few_shot.n_samples` configures the few-shot training/adaptation or prompt
demonstration condition (MI is reported at `n=5` in the paper). It does **not**
control the held-out examples used to compute DepthRank. DepthRank evaluation is
configured under `evaluation.depthrank`, where `heads_per_relation` selects held-out
heads per relation and `max_reference_tails` limits the number of gold tails retained
for each selected head.

The reproduction setup uses T5-base, the selected nine ATOMIC2020 relations,
`few_shot.n_samples: 5`, `evaluation.depthrank.heads_per_relation: 100`, and
`evaluation.depthrank.max_reference_tails: 3`. Relation selection must be explicit:
use `relations.mode: selected` with a `selected` list, or `relations.mode: all` for
extension experiments.

`mi_scores.csv` records both sides of this split: `evaluation_size`,
`few_shot_size`, `number_of_heads`, `number_of_reference_tails`, `model`, and
`seed`, in addition to relation-level DepthRank and MI fields. The runner logs a
warning if the configured evaluation head count equals the few-shot size and reports
missing selected relations plus actual evaluated head/tail counts per relation.
