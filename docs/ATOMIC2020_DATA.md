# ATOMIC2020 dataset setup

The dataset subsystem loads ATOMIC2020 without deprecated HuggingFace dataset scripts. It automatically chooses the first usable backend in this order:

1. A local converted dataset at `data/atomic2020` (preferred).
2. A HuggingFace `datasets` dataset identifier such as `allenai/atomic2020`, when it can be loaded by the installed `datasets` package without deprecated scripts.

No downstream code needs to change: callers continue to use `load_atomic2020_instances(...)` and receive `RelationInstance` objects.

## Recommended local layout

Create `data/atomic2020` with one split file per split:

```text
data/atomic2020/
  train.jsonl
  validation.jsonl
  test.jsonl
```

CSV, JSON, JSONL, and Parquet split files are supported. A HuggingFace dataset saved with `DatasetDict.save_to_disk("data/atomic2020")` is also supported.

Each row may use either a long schema:

```json
{"id": "row-1", "head": "PersonX bakes a cake", "relation": "xNeed", "tail": "buy ingredients"}
```

or a wide ATOMIC-style schema:

```json
{"id": "row-1", "event": "PersonX bakes a cake", "xNeed": ["buy ingredients"], "xIntent": ["make dessert"]}
```

Accepted head columns are `head`, `event`, `Event`, `source`, `input`, and `head_event`. Wide-schema columns other than IDs and head columns are interpreted as relation names.

## Obtaining ATOMIC2020

ATOMIC2020 is distributed by the original authors. Obtain it from the official release channel described by the ATOMIC2020 paper/project page, follow any license or access requirements, and convert the released files into the split layout above. Do not pin or downgrade `datasets`; this project intentionally avoids deprecated script-backed dataset loading.

## Verifying the installation

Run the integration test after placing the converted files in `data/atomic2020`:

```bash
pytest tests/integration/test_atomic2020_dataset.py
```

If the directory is absent, the test skips gracefully. If present, it loads the real training split and verifies normalized relation triples are non-empty.

## Configuration

Experiment configs should use the automatic backend:

```yaml
dataset:
  backend: auto
  name: atomic2020
  local_path: data/atomic2020
  hf_path: allenai/atomic2020
  cache_dir: data/cache
  split: validation
```

Use `backend: local` to require local converted files, or `backend: hf` to require a HuggingFace dataset.
