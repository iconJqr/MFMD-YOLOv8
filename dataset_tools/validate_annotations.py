"""Validate YOLO labels, image-label pairing, class ids, and box ranges."""

from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_split(root: Path, split: str, classes: int) -> list[str]:
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    errors = []
    images = {p.stem for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS} if image_dir.exists() else set()
    labels = {p.stem for p in label_dir.glob("*.txt")} if label_dir.exists() else set()
    errors.extend(f"{split}: missing label for {stem}" for stem in sorted(images - labels))
    errors.extend(f"{split}: missing image for {stem}" for stem in sorted(labels - images))
    for path in sorted(label_dir.glob("*.txt")) if label_dir.exists() else []:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                errors.append(f"{path}:{line_no}: expected 5 fields")
                continue
            try:
                cls, *values = map(float, fields)
            except ValueError:
                errors.append(f"{path}:{line_no}: non-numeric value")
                continue
            if cls != int(cls) or not 0 <= int(cls) < classes:
                errors.append(f"{path}:{line_no}: invalid class {cls}")
            if any(value < 0 or value > 1 for value in values) or values[2] <= 0 or values[3] <= 0:
                errors.append(f"{path}:{line_no}: invalid normalized box")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=3)
    args = parser.parse_args()
    errors = [error for split in ("train", "val", "test") for error in check_split(args.dataset, split, args.classes)]
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("annotation validation passed")


if __name__ == "__main__":
    main()
