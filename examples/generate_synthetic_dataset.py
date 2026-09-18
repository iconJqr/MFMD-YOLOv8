"""Generate non-research toy images for installation and pipeline smoke tests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


COLORS = ((70, 100, 220), (80, 190, 100), (220, 140, 60))


def generate(root: Path, size: int = 640, seed: int = 42) -> Path:
    rng = random.Random(seed)
    counts = {"train": 8, "val": 2, "test": 2}
    for split, count in counts.items():
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = np.full((size, size, 3), 225, dtype=np.uint8)
            cls = (index + {"train": 0, "val": 1, "test": 2}[split]) % 3
            width, height = rng.randint(size // 5, size // 3), rng.randint(size // 5, size // 3)
            left, top = rng.randint(5, size - width - 5), rng.randint(5, size - height - 5)
            cv2.rectangle(image, (left, top), (left + width, top + height), COLORS[cls], -1)
            stem = f"synthetic_{split}_{index:03d}"
            cv2.imwrite(str(image_dir / f"{stem}.png"), image)
            xc, yc = (left + width / 2) / size, (top + height / 2) / size
            (label_dir / f"{stem}.txt").write_text(
                f"{cls} {xc:.8f} {yc:.8f} {width/size:.8f} {height/size:.8f}\n", encoding="utf-8")
    data = {"path": str(root.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {0: "geometry", 1: "animal", 2: "plant"},
            "provenance": "synthetic_smoke_test_only"}
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (root / "README.txt").write_text(
        "Synthetic colored rectangles for software smoke tests only. Not research data and not suitable for reporting.\n",
        encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("synthetic_dataset"))
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(generate(args.output, args.size, args.seed))


if __name__ == "__main__":
    main()
