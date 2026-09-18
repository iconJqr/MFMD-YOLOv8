"""Evaluate fused class-specific Grad-CAM maps with the Pointing Game."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PATTERN = re.compile(r"^(?P<stem>.+)_class(?P<class_id>\d+)_fusion\.npy$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmaps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("pointing_game.csv"))
    args = parser.parse_args()
    counts = defaultdict(lambda: [0, 0])
    for path in sorted(args.heatmaps.glob("*_fusion.npy")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        cls = int(match.group("class_id"))
        boxes = []
        label_file = args.labels / f"{match.group('stem')}.txt"
        if not label_file.exists():
            continue
        for line in label_file.read_text(encoding="utf-8").splitlines():
            values = list(map(float, line.split()))
            if int(values[0]) == cls:
                boxes.append(values[1:])
        if not boxes:
            continue
        heatmap = np.load(path)
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        xn, yn = (x + 0.5) / heatmap.shape[1], (y + 0.5) / heatmap.shape[0]
        hit = any(abs(xn - xc) <= width / 2 and abs(yn - yc) <= height / 2 for xc, yc, width, height in boxes)
        counts[cls][0] += int(hit)
        counts[cls][1] += 1
    rows = [{"class_id": cls, "hits": hit, "evaluated": total, "accuracy": hit / total}
            for cls, (hit, total) in sorted(counts.items())]
    if rows:
        rows.append({"class_id": "macro_average", "hits": "", "evaluated": "",
                     "accuracy": sum(row["accuracy"] for row in rows) / len(rows)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("class_id", "hits", "evaluated", "accuracy"))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
