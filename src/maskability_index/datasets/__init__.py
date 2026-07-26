"""Dataset loading public API and configuration utilities."""

from .manager import DatasetConfig
from .atomic import (
    ATOMIC2020_HF_PATH,
    RelationInstance,
    canonical_split_name,
    iter_relation_instances,
    load_atomic2020_dataset,
    load_atomic2020_instances,
    sample_heads_per_relation,
    sample_instances_per_relation,
)

# Keep loader helper import for backwards compat if other code imports it
from .loader import load_hf_dataset

__all__ = [
    "DatasetConfig",
    "ATOMIC2020_HF_PATH",
    "RelationInstance",
    "canonical_split_name",
    "iter_relation_instances",
    "load_atomic2020_dataset",
    "load_atomic2020_instances",
    "sample_heads_per_relation",
    "sample_instances_per_relation",
    "load_hf_dataset",
]
