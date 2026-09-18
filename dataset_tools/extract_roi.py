"""Extract a manually specified ROI and convert it to the paper's 640 canvas."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--box", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--margin", type=float, default=0.10, help="Symmetric ROI expansion, 0.10 to 0.15")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.10 <= args.margin <= 0.15:
        raise ValueError("--margin must be between 0.10 and 0.15")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = args.box
    roi_w, roi_h = x2 - x1, y2 - y1
    x1 -= roi_w * args.margin / 2
    x2 += roi_w * args.margin / 2
    y1 -= roi_h * args.margin / 2
    y2 += roi_h * args.margin / 2
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("ROI is empty after clipping")
    scale = 640 / max(crop.shape[:2])
    new_w, new_h = round(crop.shape[1] * scale), round(crop.shape[0] * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=resized.dtype)
    left, top = (640 - new_w) // 2, (640 - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise IOError(f"Could not write {args.output}")


if __name__ == "__main__":
    main()
