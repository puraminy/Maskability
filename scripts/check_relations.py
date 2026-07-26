#!/usr/bin/env python3
"""
CLI wrapper for the relation-coverage diagnostics.

Usage:
  python scripts/check_relations.py --local-path data/atomic
  python scripts/check_relations.py --local-path data/atomic2020 --backend auto
"""
import argparse
import json
from pathlib import Path

from maskability_index.templates.diagnostics import find_unknown_relations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-path", required=True, help="Path to local dataset directory (data/atomic or data/atomic2020)")
    p.add_argument("--backend", default="local", choices=["local", "auto", "hf", "csv"], help="Loader backend to use when scanning (default: local)")
    p.add_argument("--hf-path", default=None, help="Optional HF id to use when backend=auto/hf")
    p.add_argument("--cache-dir", default=None, help="Optional cache directory for HF datasets")
    p.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = p.parse_args()

    report = find_unknown_relations(
        local_path=Path(args.local_path),
        backend=args.backend,
        hf_path=args.hf_path,
        cache_dir=args.cache_dir,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("Dataset relation coverage diagnostics")
    print("===================================")
    print(f"scanned splits: {report['splits']}")
    print(f"total instances: {report['total_instances']}")
    print()
    print("Registered relations (count):")
    for r in sorted(report["registered_relations"]):
        print("  ", r)
    print()
    print("Detected relations and counts (top 50):")
    for rel, count in sorted(report["counts"].items(), key=lambda t: -t[1])[:50]:
        sample = ", ".join(report["samples"].get(rel, [])[:3])
        print(f"  {rel:20s} {count:8d}   sample heads: {sample!r}")
    print()
    if report["unknown_relations"]:
        print("Unknown relations (no registered template):")
        for rel in sorted(report["unknown_relations"]):
            print("  ", rel)
    else:
        print("All detected relations are covered by the template registry.")
    print()
    print("If unknown relations are present, consider mapping them to canonical names or registering templates.")
    print("Example mapping in loader (atomic.py): RELATION_ALIASES = {'output': 'oEffect'}")


if __name__ == "__main__":
    main()
