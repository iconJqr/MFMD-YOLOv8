"""Reconstruct a deterministic source-group-aware train/val/test allocation.

This is a replacement implementation based on the manuscript, not the lost
original split script. An existing split column is never silently overwritten.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


FEATURES = ("image_count", "geometry_count", "animal_count", "plant_count")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "source_group_id" not in rows[0]:
        raise ValueError("manifest must contain source_group_id")
    if any(row.get("split", "").strip() for row in rows):
        raise ValueError("input already contains split assignments; clear them explicitly before reconstruction")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_group_id"]].append(row)
    vectors = {}
    for group, items in grouped.items():
        vectors[group] = {
            "image_count": len(items),
            **{name: sum(int(item.get(name, 0) or 0) for item in items) for name in FEATURES[1:]},
        }
    totals = {name: sum(vector[name] for vector in vectors.values()) for name in FEATURES}
    targets = {split: {name: totals[name] * ratio for name in FEATURES} for split, ratio in RATIOS.items()}
    assigned = {split: {name: 0 for name in FEATURES} for split in RATIOS}
    allocations = {}
    randomizer = random.Random(args.seed)
    groups = list(grouped)
    randomizer.shuffle(groups)
    groups.sort(key=lambda group: sum(vectors[group][name] / max(totals[name], 1) for name in FEATURES), reverse=True)

    def objective(split: str, vector: dict[str, int]) -> float:
        score = 0.0
        for name in FEATURES:
            future = assigned[split][name] + vector[name]
            score += ((future - targets[split][name]) / max(targets[split][name], 1)) ** 2
        return score

    for group in groups:
        split = min(RATIOS, key=lambda candidate: (objective(candidate, vectors[group]), assigned[candidate]["image_count"]))
        allocations[group] = split
        for name in FEATURES:
            assigned[split][name] += vectors[group][name]
    for row in rows:
        row["split"] = allocations[row["source_group_id"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split, values in assigned.items():
        print(split, values)


if __name__ == "__main__":
    main()
