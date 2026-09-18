"""Plot P/R/mAP training curves from real Ultralytics results.csv files.

This utility never synthesizes missing epochs. Each legend entry must be paired
with a retained results.csv using ``--run NAME=PATH``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "precision": ("metrics/precision(B)", "metrics/precision", "P"),
    "recall": ("metrics/recall(B)", "metrics/recall", "R"),
    "map50": ("metrics/mAP50(B)", "metrics/mAP_0.5", "mAP@0.5"),
    "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95", "mAP@0.5:0.95"),
}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/results.csv")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser()


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    normalized = {str(column).strip(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise KeyError(f"None of {candidates} found; available columns: {list(normalized)}")


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME=results.csv; repeatable")
    parser.add_argument("--output", type=Path, default=Path("training_curves"))
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--aspect", choices=("4:3", "5:4"), default="4:3")
    parser.add_argument("--raw-alpha", type=float, default=0.18)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    size = (8.0, 6.0) if args.aspect == "4:3" else (7.5, 6.0)
    colors = plt.get_cmap("tab20").colors
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, path in args.run:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        epoch_column = find_column(frame, ("epoch",))
        frames.append((name, frame, epoch_column))

    for key, candidates in METRICS.items():
        fig, axis = plt.subplots(figsize=size, constrained_layout=True)
        for index, (name, frame, epoch_column) in enumerate(frames):
            column = find_column(frame, candidates[:2])
            epoch = pd.to_numeric(frame[epoch_column], errors="coerce").to_numpy()
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            valid = np.isfinite(epoch) & np.isfinite(values)
            epoch, values = epoch[valid], values[valid]
            color = colors[index % len(colors)]
            axis.plot(epoch, values, color=color, alpha=args.raw_alpha, linewidth=1.2)
            axis.plot(epoch, smooth(values, args.smooth_window), color=color, linewidth=2.4, label=name)
        axis.set_xlabel("Epoch", fontsize=14, fontweight="bold")
        axis.set_ylabel(candidates[2], fontsize=14, fontweight="bold")
        axis.tick_params(labelsize=12, width=1.2)
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            label.set_fontweight("bold")
        axis.grid(alpha=0.22, linewidth=0.7)
        legend = axis.legend(loc="lower right", fontsize=11, framealpha=0.92, handlelength=3.4)
        for text in legend.get_texts():
            text.set_fontweight("bold")
        for line in legend.get_lines():
            line.set_linewidth(3.0)
        for suffix in ("png", "svg"):
            fig.savefig(args.output / f"{key}.{suffix}", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
