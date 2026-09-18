"""Aggregate per-seed comparison metrics using sample standard deviation."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("precision", "recall", "map50", "map50_95", "params_m", "gflops", "fps")
EXPECTED_SEEDS = {0, 1, 42}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("model"):
                grouped[row["model"]].append(row)

    output_rows = []
    for model, rows in sorted(grouped.items()):
        seeds = {int(row["seed"]) for row in rows}
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"{model}: expected seeds {sorted(EXPECTED_SEEDS)}, found {sorted(seeds)}")
        summary = {"model": model, "runs": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric, "").strip()]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else math.nan
            summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else math.nan
        output_rows.append(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "runs", *(item for metric in METRICS for item in (f"{metric}_mean", f"{metric}_sd"))]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
