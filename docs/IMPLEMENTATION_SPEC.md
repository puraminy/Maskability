# Implementation Specification

## Purpose

This document translates the paper into implementation requirements.

The manuscript is the scientific specification.

The implementation must reproduce the methodology exactly.

If any equation, algorithm, or experimental detail is ambiguous, create `docs/AMBIGUITIES.md` instead of making assumptions.

---

# Module 1: Dataset

Input:
- ATOMIC2020

Output:
- head
- relation
- tail

Requirements:
- Download automatically using HuggingFace Datasets.
- Cache locally.
- Preserve original relation names.
- Support train/dev/test splits.

---

# Module 2: Template Generator

Input:
(head, relation, tail)

Output:
- Prefix prompt
- Masked prompt

Requirements:
- Implement templates exactly as described in the paper.
- One canonical template per relation.
- Support future alternative templates.

---

# Module 3: Prompt Builder

Input:
Relation instance

Output:
Training example

Responsibilities:
- n-shot prompting
- Prompt ordering
- Demonstration selection
- Masked prompting
- Prefix prompting

No hardcoded prompts.

---

# Module 4: Training

Backend:
HuggingFace Transformers

Requirements:
- Seq2SeqTrainer
- fp16 support
- checkpointing
- deterministic seeds
- configurable hyperparameters

Supported models:
- google/t5-small
- google/t5-base
- google/t5-large

---

# Module 5: Inference

Generate predictions.

Support:
- greedy
- beam search

Store:
- prediction
- score
- probability (if available)

---

# Module 6: DepthRank

Implement exactly according to the manuscript.

Do not simplify.

One function per equation.

Document correspondence between equations and code.

---

# Module 7: Maskability Index

Implement MI exactly according to the manuscript.

Input:
DepthRank(masked)
DepthRank(prefix)

Output:
MI

No heuristic modifications.

---

# Module 8: Statistics

Implement

- Pearson
- Spearman
- Kendall
- Bootstrap CI
- Permutation test

---

# Module 9: Figures

Automatically generate

- scatter plots
- threshold plots
- sensitivity plots
- correlation plots

---

# Module 10: Outputs

Every experiment must produce

metrics.json

config.yaml

predictions.csv

plots/

latex/

log.txt

checkpoint/

---

# Module 11: Tests

Unit tests required for

Template generation

Prompt generation

DepthRank

MI

Statistics

---

# Module 12: Reproducibility

Every experiment stores

Git commit

Seed

Model version

Dataset version

Configuration

Execution time

Hardware information

