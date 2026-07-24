# Maskability Index Framework

Infrastructure for reproducing the experiments from “The Maskability Index: Predicting Task–Objective Alignment in Pretrained Language Models”.

Milestone 1 provides only project infrastructure: configuration, logging, tracking, output directories, reproducibility metadata, and test scaffolding. Scientific algorithms, training, DepthRank, and MI are intentionally not implemented yet.

## Quick start

```bash
uv sync --extra dev
uv run maskability-run
uv run pytest
```

Hydra configurations live in `configs/`; runtime outputs are created under `results/`.


## ATOMIC2020 data

See `docs/DATASET.md` for instructions to obtain and place the official ATOMIC2020 CSV files.
