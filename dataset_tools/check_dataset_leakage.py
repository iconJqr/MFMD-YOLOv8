"""Check that source groups do not occur in multiple dataset subsets."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-column", default="source_group_id")
    parser.add_argument("--split-column", default="split")
    args = parser.parse_args()
    groups: dict[str, set[str]] = defaultdict(set)
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            groups[row[args.source_column]].add(row[args.split_column])
    conflicts = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    if conflicts:
        for group, splits in conflicts.items():
            print(f"{group}: {','.join(splits)}")
        raise SystemExit(1)
    print(f"source-group leakage check passed for {len(groups)} groups")


if __name__ == "__main__":
    main()
