"""Compare completed Maskability experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from maskability_index.evaluation import compare_runs, export_comparison


def main() -> None:
    """Write comparison.csv, comparison.tex, and summary.md for completed runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Completed experiment result directories.")
    parser.add_argument("--output-dir", default=".", help="Directory for comparison artifacts.")
    args = parser.parse_args()
    export_comparison(compare_runs([Path(run) for run in args.runs]), args.output_dir)


if __name__ == "__main__":
    main()
