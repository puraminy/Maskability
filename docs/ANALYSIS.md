# Analysis and Evaluation Layer

The analysis layer consumes completed experiment outputs and does not introduce new scientific methods. It aggregates values already produced by the experiment engine: DepthRank, relation-level Maskability Index (MI), correlations, bootstrap confidence intervals, and permutation tests.

## Evaluation

`maskability_index.evaluation.evaluator` loads a completed run directory containing `metrics.json`, `mi_scores.csv`, `depthrank.csv`, and `predictions.csv`. It computes run-level summaries such as mean MI, best MI, best relation, mean DepthRank, best threshold when present, and best n-shot when present.

Every completed experiment is indexed in `results/index.json`. Index entries store the experiment ID, timestamp, Git commit, model, dataset, configuration hash, aggregate metrics, and paths to standard artifacts.

## Comparison

Use the comparison script with one or more completed run directories:

```bash
python experiments/compare_runs.py results/E01 results/E02 results/E03
```

The script writes:

- `comparison.csv`
- `comparison.tex`
- `summary.md`

Runs are ranked by mean MI. Relative improvement is computed against the lowest ranked run in the comparison table, using the standard relative difference formula rather than any new scientific metric.

## Tables

Each experiment writes LaTeX tables under its `latex/` directory:

- `table_1.tex`: relation-level MI results
- `table_2.tex`: maskability group summary
- `tables.tex`: combined tables
- `table_1.csv` and `table_1.md`: machine-readable and Markdown versions

These tables are generated directly from experiment CSV outputs.

## Figures

`maskability_index.plotting.generate_plots` creates publication-oriented figures in PNG, PDF, and SVG formats:

- scatter
- histogram
- correlation
- threshold sensitivity
- n-shot sensitivity
- model comparison
- baseline comparison

The plotting module only visualizes columns already present in CSV or JSON outputs; it does not compute scientific metrics.

## Reports

Generate a summary report with:

```bash
python experiments/generate_report.py results/E01 results/E02 --output-dir results/report
```

The report writer produces:

- `results/report/summary.md`
- `results/report/summary.tex`
- regenerated figures in `results/report/plots/`

The report summarizes best run/model, best threshold, best n-shot, best MI, correlations, confidence intervals, and plots when those fields are available in experiment outputs. PDF report creation is optional and intentionally left to the user’s LaTeX environment.
