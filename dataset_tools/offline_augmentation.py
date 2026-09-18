"""Reconstructed training-only offline augmentation exporter.

The parameter ranges come from the manuscript. The lost original operation
ordering and RandAugment magnitude mapping cannot be recovered, so this script
records its own deterministic choices in an audit CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_boxes(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            cls, xc, yc, bw, bh = map(float, line.split())
            boxes.append([cls, (xc - bw / 2) * width, (yc - bh / 2) * height,
                          (xc + bw / 2) * width, (yc + bh / 2) * height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 5)


def save_boxes(path: Path, boxes: np.ndarray, width: int, height: int) -> None:
    rows = []
    for cls, x1, y1, x2, y2 in boxes:
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(f"{int(cls)} {(x1 + x2) / 2 / width:.8f} {(y1 + y2) / 2 / height:.8f} "
                    f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}")
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def affine(image: np.ndarray, boxes: np.ndarray, rng: random.Random, translate: float, scale_range: list[float], shear=0.0):
    height, width = image.shape[:2]
    scale = rng.uniform(*scale_range)
    tx, ty = rng.uniform(-translate, translate) * width, rng.uniform(-translate, translate) * height
    shear_value = math.tan(math.radians(rng.uniform(-shear, shear))) if shear else 0.0
    matrix = np.array([[scale, shear_value, tx + (1 - scale) * width / 2],
                       [0, scale, ty + (1 - scale) * height / 2]], dtype=np.float32)
    output = cv2.warpAffine(image, matrix, (width, height), borderValue=(114, 114, 114))
    if len(boxes):
        corners = np.stack([boxes[:, [1, 2]], boxes[:, [3, 2]], boxes[:, [3, 4]], boxes[:, [1, 4]]], axis=1)
        transformed = corners @ matrix[:, :2].T + matrix[:, 2]
        boxes[:, 1] = transformed[:, :, 0].min(1)
        boxes[:, 2] = transformed[:, :, 1].min(1)
        boxes[:, 3] = transformed[:, :, 0].max(1)
        boxes[:, 4] = transformed[:, :, 1].max(1)
    return output, boxes


def filter_boxes(before: np.ndarray, after: np.ndarray, width=640, height=640) -> np.ndarray:
    if not len(after):
        return after
    clipped = after.copy()
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, width)
    clipped[:, [2, 4]] = clipped[:, [2, 4]].clip(0, height)
    old_area = np.maximum(before[:, 3] - before[:, 1], 0) * np.maximum(before[:, 4] - before[:, 2], 0)
    new_w, new_h = clipped[:, 3] - clipped[:, 1], clipped[:, 4] - clipped[:, 2]
    keep = (new_w >= 10) & (new_h >= 10) & ((new_w * new_h) / np.maximum(old_area, 1) >= 0.20)
    return clipped[keep]


def hsv(image: np.ndarray, rng: random.Random) -> np.ndarray:
    array = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    array[:, :, 0] = (array[:, :, 0] + rng.uniform(-14, 14) / 2) % 180
    array[:, :, 1] *= rng.uniform(0.75, 1.25)
    array[:, :, 2] *= rng.uniform(0.80, 1.20)
    return cv2.cvtColor(np.clip(array, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_erasing(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    target = rng.uniform(0.02, 0.15) * height * width
    ratio = rng.uniform(0.35, 2.80)
    erase_w = min(width, max(1, round(math.sqrt(target * ratio))))
    erase_h = min(height, max(1, round(math.sqrt(target / ratio))))
    left, top = rng.randint(0, width - erase_w), rng.randint(0, height - erase_h)
    image[top:top + erase_h, left:left + erase_w] = image.mean(axis=(0, 1), dtype=np.float64)
    return image


def mosaic(primary: Path, candidates: list[Path], dataset: Path, rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construct a deterministic four-image reconstructed mosaic on a 640 canvas."""
    selected = [primary, *rng.sample(candidates, 3)]
    cx, cy = round(rng.uniform(0.38, 0.62) * 640), round(rng.uniform(0.38, 0.62) * 640)
    regions = ((0, 0, cx, cy), (cx, 0, 640, cy), (0, cy, cx, 640), (cx, cy, 640, 640))
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    output_boxes = []
    for path, (x1, y1, x2, y2) in zip(selected, regions):
        source = cv2.imread(str(path))
        if source is None:
            raise ValueError(f"Unreadable image: {path}")
        source_boxes = load_boxes(dataset / "labels" / "train" / f"{path.stem}.txt", source.shape[1], source.shape[0])
        region_w, region_h = x2 - x1, y2 - y1
        scale = max(region_w / source.shape[1], region_h / source.shape[0])
        resized = cv2.resize(source, (round(source.shape[1] * scale), round(source.shape[0] * scale)))
        crop_left, crop_top = max(0, (resized.shape[1] - region_w) // 2), max(0, (resized.shape[0] - region_h) // 2)
        canvas[y1:y2, x1:x2] = resized[crop_top:crop_top + region_h, crop_left:crop_left + region_w]
        if len(source_boxes):
            before = source_boxes.copy()
            source_boxes[:, [1, 3]] = source_boxes[:, [1, 3]] * scale - crop_left + x1
            source_boxes[:, [2, 4]] = source_boxes[:, [2, 4]] * scale - crop_top + y1
            source_boxes[:, [1, 3]] = source_boxes[:, [1, 3]].clip(x1, x2)
            source_boxes[:, [2, 4]] = source_boxes[:, [2, 4]].clip(y1, y2)
            scaled_before = before.copy()
            scaled_before[:, [1, 3]] *= scale
            scaled_before[:, [2, 4]] *= scale
            output_boxes.append(filter_boxes(scaled_before, source_boxes))
    boxes = np.concatenate(output_boxes) if output_boxes else np.empty((0, 5), dtype=np.float32)
    return canvas, boxes, [path.name for path in selected]


def augment(image: np.ndarray, boxes: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[str]]:
    operations = []
    if rng.random() < 0.80:
        image, operations = hsv(image, rng), operations + ["hsv"]
    if rng.random() < 0.60:
        before = boxes.copy()
        image, boxes = affine(image, boxes, rng, 0.12, [0.78, 1.22])
        boxes, operations = filter_boxes(before, boxes), operations + ["translation_scaling"]
    if rng.random() < 0.50:
        image = cv2.flip(image, 1)
        if len(boxes):
            old_x1, old_x2 = boxes[:, 1].copy(), boxes[:, 3].copy()
            boxes[:, 1], boxes[:, 3] = image.shape[1] - old_x2, image.shape[1] - old_x1
        operations.append("horizontal_flip")
    if rng.random() < 0.50:
        choices = rng.sample(["brightness", "contrast", "sharpness", "shear"], 2)
        for choice in choices:
            if choice == "brightness":
                image = cv2.convertScaleAbs(image, alpha=rng.uniform(0.75, 1.25), beta=0)
            elif choice == "contrast":
                image = cv2.convertScaleAbs(image, alpha=rng.uniform(0.75, 1.25), beta=rng.uniform(-20, 20))
            elif choice == "sharpness":
                blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
                image = cv2.addWeighted(image, rng.uniform(1.0, 1.6), blurred, rng.uniform(-0.6, 0.0), 0)
            else:
                before = boxes.copy()
                image, boxes = affine(image, boxes, rng, 0.0, [1.0, 1.0], shear=7.0)
                boxes = filter_boxes(before, boxes)
        operations.append("randaugment:" + "+".join(choices))
    if rng.random() < 0.30:
        image, operations = random_erasing(image, rng), operations + ["random_erasing"]
    return image, boxes, operations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Root containing images/train and labels/train")
    parser.add_argument("--output", type=Path, required=True, help="A new output dataset root")
    parser.add_argument("--config", type=Path, default=Path("configs/offline_augmentation_reconstructed.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rng = random.Random(config["seed"])
    source_images = sorted(p for p in (args.dataset / "images" / "train").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    image_out, label_out = args.output / "images" / "train", args.output / "labels" / "train"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    audit = []
    for source in source_images:
        label = args.dataset / "labels" / "train" / f"{source.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(label)
        shutil.copy2(source, image_out / source.name)
        shutil.copy2(label, label_out / label.name)
        base = cv2.imread(str(source))
        if base is None:
            raise ValueError(f"Unreadable image: {source}")
        if base.shape[:2] != (640, 640):
            raise ValueError(f"Expected prepared 640x640 image: {source}")
        original_boxes = load_boxes(label, 640, 640)
        for index in range(1, int(config["outputs_per_original"]) + 1):
            if rng.random() < config["operations"]["mosaic"]["probability"] and len(source_images) >= 4:
                others = [path for path in source_images if path != source]
                image, boxes, mosaic_sources = mosaic(source, others, args.dataset, rng)
                prefix = ["mosaic:" + "+".join(mosaic_sources)]
            else:
                image, boxes, prefix = base.copy(), original_boxes.copy(), []
            image, boxes, operations = augment(image, boxes, rng)
            operations = prefix + operations
            stem = f"{source.stem}_aug{index:02d}"
            cv2.imwrite(str(image_out / f"{stem}.png"), image)
            save_boxes(label_out / f"{stem}.txt", boxes, 640, 640)
            audit.append({"source": source.name, "output": f"{stem}.png", "operations": "|".join(operations)})
    with (args.output / "augmentation_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source", "output", "operations"))
        writer.writeheader()
        writer.writerows(audit)
    print(f"original={len(source_images)} augmented={len(audit)} total={len(source_images) + len(audit)}")


if __name__ == "__main__":
    main()
