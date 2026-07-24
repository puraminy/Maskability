# Maskability Index Framework
Version: 1.0

---

# 1. Goal

This repository implements a fully reproducible research framework for the paper

"The Maskability Index: Predicting Task–Objective Alignment in Pretrained Language Models"

The implementation is intended to reproduce the published experiments and support additional experiments requested during peer review.

Scientific correctness has higher priority than engineering convenience.

If the paper is ambiguous, the implementation must stop and document the ambiguity instead of making assumptions.

---

# 2. General Principles

The framework must satisfy

✓ reproducibility

✓ modularity

✓ extensibility

✓ deterministic execution

✓ experiment tracking

No hardcoded paths.

No hardcoded datasets.

No hardcoded prompt templates.

No hardcoded thresholds.

Everything configurable.

---

# 3. Architecture

                Dataset

                   │

                   ▼

        Relation Template Generator

                   │

         ┌─────────┴─────────┐

         ▼                   ▼

 Prefix Prompt        Masked Prompt

         └─────────┬─────────┘

                   ▼

          Dataset Construction

                   ▼

             Fine-tuning

                   ▼

          DepthRank Evaluation

                   ▼

        Maskability Index (MI)

                   ▼

      Statistics / Visualization

---

# 4. Repository Structure

src/

    datasets/

    prompting/

    templates/

    models/

    training/

    evaluation/

    depthrank/

    maskability/

    statistics/

    plotting/

    utils/

configs/

experiments/

results/

tests/

docs/

---

# 5. Dataset Module

Responsibilities

- download datasets

- cache datasets

- preprocess

- expose a unified API

Initial implementation

ATOMIC2020

Future datasets should require no changes elsewhere.

---

# 6. Template Module

Responsibilities

Convert relation types into textual templates.

Support

- Prefix prompting

- Masked prompting

- Future prompting styles

Prompt templates must not be embedded elsewhere.

---

# 7. Prompt Module

Given

(head, relation, tail)

generate

training instances

evaluation instances

few-shot demonstrations

Support configurable n-shot prompting.

---

# 8. Training Module

Responsibilities

Fine-tune Seq2Seq language models.

Initial backend

HuggingFace Transformers

Use Trainer API unless a research requirement makes a custom loop necessary.

---

# 9. Model Module

Must support

T5-small

T5-base

T5-large

Future models should require only configuration changes.

---

# 10. Evaluation Module

Responsibilities

Generate predictions

Compute ranking metrics

Compute DepthRank

Return structured outputs

No plotting.

No statistics.

---

# 11. DepthRank Module

Implement exactly the algorithm described in the paper.

The implementation must map every equation to one function.

Every variable should correspond to the notation used in the manuscript.

No approximations.

---

# 12. Maskability Module

Implement the Maskability Index exactly as defined in the paper.

No implicit normalization.

No heuristic modifications.

The implementation must remain mathematically traceable to the manuscript.

---

# 13. Statistics Module

Provide reusable functions for

Pearson correlation

Spearman correlation

Kendall correlation

Bootstrap confidence intervals

Permutation tests

Future statistical analyses should be added here.

---

# 14. Plotting Module

Responsible only for visualization.

Never compute metrics.

Input

CSV

JSON

Output

PNG

PDF

SVG

---

# 15. Configuration

Every experiment must be executable from YAML.

Nothing scientific should be hardcoded.

Example

model

learning rate

batch size

epochs

threshold

n-shot

prompt style

dataset

seed

---

# 16. Experiment Tracking

Every experiment automatically stores

configuration

random seed

Git commit hash

metrics

plots

logs

checkpoints

---

# 17. Testing

Each scientific component must have unit tests.

Template generation

DepthRank

MI computation

Statistics

---

# 18. Extensibility

Reviewer-requested experiments should require only

new configuration files

or

new experiment scripts

The scientific framework should never require modification.
