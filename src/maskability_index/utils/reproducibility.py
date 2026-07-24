"""Reproducibility helpers for deterministic experiment execution."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class EnvironmentInfo:
    """Serializable execution environment metadata."""

    python: str
    platform: str
    torch: str
    cuda_available: bool
    git_hash: str
    cwd: str


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_hash(repo_root: Path | str = ".") -> str:
    """Return the current Git commit hash, or `unknown` outside a Git checkout."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect_environment_info(repo_root: Path | str = ".") -> dict[str, Any]:
    """Collect deterministic-run metadata for storage with experiment outputs."""

    info = EnvironmentInfo(
        python=sys.version.replace("\n", " "),
        platform=platform.platform(),
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        git_hash=git_hash(repo_root),
        cwd=str(Path.cwd()),
    )
    return asdict(info)
