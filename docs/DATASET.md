# ATOMIC2020 dataset

This project loads ATOMIC2020 from the official CSV files released by the dataset authors. The loader does **not** download data automatically.

## Download source

Download ATOMIC2020 from the official ATOMIC release URL, <https://maartensap.com/atomic/data/atomic_data.tgz>, following the license terms included with the distribution. The expected archive contains files such as `README.md`, `LICENSE`, `sap2019atomic.pdf`, `v4_atomic_all.csv`, `v4_atomic_all_agg.csv`, `v4_atomic_trn.csv`, `v4_atomic_dev.csv`, and `v4_atomic_tst.csv`.

## Expected directory structure

Extract or copy the split CSV files into the repository under:

```text
data/atomic/
  v4_atomic_trn.csv
  v4_atomic_dev.csv
  v4_atomic_tst.csv
```

Additional files from the release may also be present in `data/atomic/`; the loader only reads the three split files above.

## Parsing assumptions

The default dataset backend reads the author-provided CSV format directly:

- The event/head text is read from the `event` column.
- All remaining ATOMIC relation columns, such as `xNeed`, `xIntent`, `xWant`, `xEffect`, `xReact`, `xAttr`, `oEffect`, `oReact`, and `oWant`, are treated as relation columns.
- Relation cells are parsed as Python-literal lists, matching the distributed CSV representation, for example `['buy food', 'turn on the stove']`.
- Empty cells, `[]`, and `none` targets are skipped.
- Every non-empty `(event, relation, target)` entry is converted into a `RelationInstance` so downstream code can keep using `load_atomic2020_instances(...)` unchanged.
- Split aliases `dev`, `valid`, and `val` map to `validation`.

## Relation-balanced sampling

Experiment configs may define `dataset.sampling` to select evaluation examples after split loading and normalization:

```yaml
dataset:
  split: validation
  sampling:
    strategy: deterministic  # deterministic | random
    instances_per_relation: 5
    seed: ${experiment.seed}
```

Sampling is applied independently within each relation, so a global truncation cannot drop later relation columns such as `xAttr`. `deterministic` preserves the official CSV order and takes the first configured examples per relation. `random` uses the configured seed through a local RNG, which supports reproducible reviewer robustness, sample-size sensitivity, and multi-seed runs. Relations with fewer available examples keep all available examples rather than being discarded.

## Relation selection and evaluation limits

Hydra `relations` config controls which loaded relations are evaluated. `mode: dataset` and `mode: all` keep every relation present in the split, while `mode: selected` keeps only exact names from `selected`. The separate `evaluation.max_instances_per_relation` cap controls the number of evaluated examples per relation after relation filtering and dataset sampling. This is intentionally separate from few-shot prompting demonstrations.

## Supported dataset version

The supported layout is the ATOMIC2020 v4 CSV release with split files named `v4_atomic_trn.csv`, `v4_atomic_dev.csv`, and `v4_atomic_tst.csv`.

## HuggingFace backend

HuggingFace support is intentionally separate. The CSV backend is the default and never downloads data. A future or explicit experiment can request `backend="hf"`, but normal production runs should use the local CSV files in `data/atomic/`.

## few-shot n != DepthRank evaluation size

Do not use the number of few-shot examples as the number of DepthRank evaluation
items. Few-shot sampling configures demonstrations/adaptation examples through
`few_shot.n_samples` and its own seed. Held-out DepthRank sampling configures
`evaluation.depthrank.heads_per_relation`, `evaluation.depthrank.max_reference_tails`,
and its own seed.

Relation selection is explicit: `relations.mode: selected` requires the named
relations, while `relations.mode: all` is reserved for extension experiments over
every loaded relation. Missing selected relations are reported rather than silently
inferred from the dataset.
