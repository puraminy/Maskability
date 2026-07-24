# Project Understanding

## 1. Repository architecture

The repository is organized as a reproducible research framework for the Maskability Index paper. The intended flow is configuration-driven and modular: datasets are normalized into relation triples, templates render relation-specific prompts, prompt builders create prefix or masked inputs, models and training use HuggingFace seq2seq abstractions, evaluation produces prediction and ranking artifacts, DepthRank scores are aggregated into Maskability Index (MI), and downstream statistics/plotting/reporting operate only on structured outputs.

Main packages under `src/maskability_index/`:

- `config/`: Hydra/OmegaConf loading and typed configuration dataclasses.
- `datasets/`: HuggingFace dataset loaders and ATOMIC2020 normalization into `RelationInstance` triples.
- `templates/`: canonical relation verbalizers and a deterministic template registry.
- `prompting/`: prefix, masked, and deterministic few-shot prompt builders that consume templates.
- `models/`: registry/factory wrappers for supported T5 seq2seq models and tokenizers.
- `training/`: dataset conversion, tokenization, generation metrics, and a thin `Seq2SeqTrainer` wrapper.
- `depthrank/`: equation-level token ranking, instance DepthRank, relation aggregation, and MI helper APIs.
- `maskability/`: MI-specific aggregation, classification, typed results, and calculator facade.
- `experiments/`: Hydra runner and experiment registry for E01-E08 reproduction/reviewer runs.
- `statistics/`: correlations, bootstrap confidence intervals, and paired permutation tests.
- `plotting/`: figure generation from MI score tables.
- `evaluation/`: completed-run loading, aggregation, comparison exports, reports, and results indexing.
- `tracking/`: MLflow-backed experiment tracking adapter.
- `utils/`: output-file, logging, git/environment, and seeding helpers.

## 2. Scientific objective

The scientific objective is to reproduce and extend the experiments from “The Maskability Index: Predicting Task–Objective Alignment in Pretrained Language Models.” The framework measures whether a relation is better aligned with masked prompting or ordinary prefix prompting by comparing mean DepthRank under the two template families. Scientific correctness is prioritized over engineering convenience, and unresolved manuscript ambiguity is supposed to be documented rather than silently guessed.

## 3. Main pipeline

The specified scientific pipeline is:

1. Load relation triples `(head, relation, tail)`, initially from ATOMIC2020.
2. Render each relation through a canonical relation template.
3. Build paired prompt families:
   - prefix prompting: unmasked continuation prompt;
   - masked prompting: prompt containing an explicit mask/span token.
4. Fine-tune or load a seq2seq model, initially T5-small/base/large.
5. Evaluate the gold tail under teacher forcing.
6. For each target token, rank the gold token within the model's next-token vocabulary distribution.
7. Average token ranks to produce instance DepthRank.
8. Average instance DepthRanks by relation and template family.
9. Compute MI as `(DR_Prompting - DR_MaskedPrompting) / DR_Prompting`.
10. Compute statistics, generate plots/tables, and save reproducibility artifacts.

## 4. Experiment pipeline

The current experiment runner creates the standard output tree, config snapshot, log file, prediction/depthrank/MI CSVs, metrics JSON, plots, LaTeX tables, a checkpoint placeholder, and a global results index. It supports synthetic rows by default and can load ATOMIC2020 when configured. The registered experiment IDs E01-E08 currently point to the same generic runner.

Important detail: the runner is presently a scaffolded/smoke-test pipeline rather than a full scientific run. It fabricates predictions as the gold tail with score/probability 1.0, and computes deterministic pseudo-DepthRank values from target length, relation name, and prompt style rather than using a trained model's teacher-forced logits.

## 5. Current implementation status

Implemented or partially implemented:

- Typed ATOMIC-style relation instances and broad support for long/wide HuggingFace dataset layouts.
- Canonical template registry and prefix/masked/few-shot prompt builders.
- T5 seq2seq model/tokenizer factory.
- Training preprocessing and `Seq2SeqTrainer` wrapper with deterministic seed setup and configurable hyperparameters.
- DepthRank equation helpers and a calculator that can score teacher-forced logits from a real seq2seq model.
- MI equation helpers, relation-level aggregation, and threshold grouping.
- Statistics, plotting, run aggregation, comparison exports, report generation, MLflow adapter, and reproducibility helpers.
- A generic Hydra experiment runner that writes the expected artifact set.

Not fully implemented in the end-to-end runner:

- Real model loading/training/inference inside the experiment pipeline.
- Real DepthRank computation from model logits in experiments.
- Few-shot demonstration selection/order policies beyond explicitly supplied demonstrations.
- Dataset/version metadata capture beyond environment/git metadata.
- Scientific validation that relation templates exactly match the manuscript.

## 6. Potential weak points

- The experiment runner's pseudo-DepthRank implementation can produce plausible-looking MI outputs that are not scientifically valid.
- Masked prompting is represented as appending a single `<extra_id_0>` span, but exact T5-style masked-target formatting and scoring details may need closer manuscript alignment.
- `rank_token` sorts the full vocabulary for each token, which is simple and traceable but may be inefficient for large vocabularies and many examples.
- Ties in token scores are not explicitly specified; Python sorting order by token id becomes the implicit tie-breaker.
- Prompt templates are labeled canonical, but their provenance and exact manuscript correspondence are not machine-checked.
- The dataset normalizer handles multiple possible schemas, but broad fallback behavior may hide upstream schema changes.
- Plotting assumes non-empty MI DataFrames with expected columns; empty or partial experiments may fail late.
- The training wrapper enforces deterministic PyTorch algorithms with `warn_only=True`, so some nondeterminism may remain at runtime.

## 7. Inconsistencies noticed

- Several module README files still say their modules are Milestone 1 placeholders, even though training, DepthRank, MI, statistics, plotting, and evaluation code now exists.
- The top-level README states that scientific algorithms, training, DepthRank, and MI are intentionally not implemented yet, which conflicts with the current source tree.
- The software specification describes repository folders like `src/datasets/`, while the actual package layout is namespaced under `src/maskability_index/`.
- The implementation specification requires every experiment to produce `checkpoint/`, but configuration defaults mention `checkpoints/`; the runner uses singular `checkpoint`.
- Documentation emphasizes that no scientific thresholds should be hardcoded, but the MI threshold default of 0.30 is embedded in `classify_maskability`, although the runner can override it via configuration.
- DepthRank/MI APIs exist both under `depthrank/` and `maskability/`, creating some duplicated aggregation and MI calculation surfaces.
- The runner imports no model/training/DepthRank calculator, despite its docstring claiming it executes training and DepthRank.
