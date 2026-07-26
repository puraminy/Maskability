#!/usr/bin/env python3

import argparse
from pathlib import Path

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", required=True)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    ds = load_from_disk(str(Path(args.local_path).expanduser()))

    print(ds)
    print()

    for split_name, split in ds.items():
        print("=" * 80)
        print(f"Split: {split_name}")
        print("=" * 80)

        print("\nColumns:")
        print(split.column_names)

        print("\nFeatures:")
        print(split.features)

        print("\nNumber of rows:")
        print(len(split))

        print("\nFirst examples:\n")

        for i in range(min(args.rows, len(split))):
            print(f"Example {i}")
            for k, v in split[i].items():
                print(f"  {k}: {repr(v)}")
            print()


if __name__ == "__main__":
    main()
