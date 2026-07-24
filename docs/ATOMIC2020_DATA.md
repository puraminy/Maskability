# ATOMIC2020 dataset setup

ATOMIC2020 data setup has moved to [`DATASET.md`](DATASET.md).

In short, download the official ATOMIC2020 v4 CSV release yourself and place the split files under `data/atomic/`:

```text
data/atomic/
  v4_atomic_trn.csv
  v4_atomic_dev.csv
  v4_atomic_tst.csv
```

The default loader reads these CSV files directly and does not download data automatically. HuggingFace loading is reserved for explicit future use via a separate backend.
