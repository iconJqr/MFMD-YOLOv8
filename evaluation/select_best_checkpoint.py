"""Select the validation-best epoch from an Ultralytics results.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    lookup = {str(value).strip(): value for value in frame.columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    raise KeyError(f"Missing columns {names}")


def select(frame: pd.DataFrame, metric: str) -> tuple[int, float]:
    epoch_col = column(frame, ("epoch",))
    map50 = pd.to_numeric(frame[column(frame, ("metrics/mAP50(B)", "metrics/mAP_0.5"))], errors="coerce")
    map95 = pd.to_numeric(frame[column(frame, ("metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95"))], errors="coerce")
    score = map95 if metric == "map50_95" else 0.1 * map50 + 0.9 * map95
    if not np.isfinite(score.to_numpy()).any():
        raise ValueError("No finite validation score")
    index = int(score.idxmax())
    return int(frame.loc[index, epoch_col]), float(score.loc[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metric", choices=("fitness", "map50_95"), default="fitness")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.results)
    epoch, score = select(frame, args.metric)
    payload = {
        "selected_epoch": epoch,
        "validation_score": score,
        "selection_metric": "0.1*mAP50+0.9*mAP50_95" if args.metric == "fitness" else "mAP50_95",
        "selection_split": "validation",
        "provenance": "computed_from_retained_results_csv",
        "note": "Use Ultralytics best.pt when its trainer used the same fitness rule.",
    }
    output = json.dumps(payload, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
