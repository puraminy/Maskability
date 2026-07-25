# Maskability Index Implementation

The Maskability Index (MI) implementation lives in `src/maskability_index/maskability/` and depends on the public DepthRank API instead of reimplementing DepthRank.

## Mathematical correspondence

For relation `r` and few-shot size `n`; held-out evaluation size is separate, the manuscript defines:

\[
MI(r,n)=\frac{DR_{Prompting}(r,n)-DR_{MaskedPrompting}(r,n)}{DR_{Prompting}(r,n)}.
\]

Implementation mapping:

- `aggregation.mean_depthrank(...)` computes each relation-level arithmetic mean DepthRank.
- `metrics.maskability_index(...)` implements the MI equation exactly.
- `MaskabilityCalculator.compute(...)` combines the two means and returns a typed result.
- `metrics.classify_maskability(...)` implements the optional paper threshold rule: `MI >= 0.30` is `Mask-Filling`; otherwise `Map-Phrasal`.

## Public API

```python
from maskability_index.maskability import MaskabilityCalculator

calculator = MaskabilityCalculator(threshold=0.30)
result = calculator.compute(
    relation="AtLocation",
    prompting=[10.0, 14.0],
    masked_prompting=[6.0, 8.0],
)
```

The result contains:

- `relation`
- `sample_size`
- `dr_prompting`
- `dr_masked_prompting`
- `maskability_index`
- `group` when a threshold is configured

## Experiment pipeline

Run:

```bash
python experiments/run_maskability.py --model google/t5-base --split validation --max-per-relation 5
```

The script:

1. Loads a trained or pretrained seq2seq model.
2. Generates masked prompts through `MaskedPromptBuilder`.
3. Generates prefix prompts through `PrefixPromptBuilder`.
4. Computes DepthRank with `DepthRankCalculator`.
5. Computes MI with `MaskabilityCalculator`.
6. Stores `mi_scores.csv`, `metrics.json`, and `config.yaml`.

Additional file `depthrank_inputs.csv` records per-instance DepthRank inputs and scores for auditability.
