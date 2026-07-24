"""Generate a publication-oriented report from completed runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from maskability_index.evaluation import generate_report


def main() -> None:
    """Write results/report/summary.md and summary.tex."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="*", default=["results/reproduction"], help="Run directories."
    )
    parser.add_argument("--output-dir", default="results/report", help="Report output directory.")
    args = parser.parse_args()
    generate_report([Path(run) for run in args.runs], args.output_dir)


if __name__ == "__main__":
    main()
