"""Evaluate original, resolution-reduced, and brightness-modified test images."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from evaluation.paper_validation import fixed_threshold_metrics, validate_paper

CONDITIONS = ("original", "resolution_480", "resolution_320", "brightness_0.6", "brightness_1.4")


def transform(image: Image.Image, condition: str) -> Image.Image:
    if condition == "resolution_480":
        return image.resize((480, 480), Image.Resampling.BILINEAR).resize((640, 640), Image.Resampling.BILINEAR)
    if condition == "resolution_320":
        return image.resize((320, 320), Image.Resampling.BILINEAR).resize((640, 640), Image.Resampling.BILINEAR)
    if condition.startswith("brightness_"):
        return ImageEnhance.Brightness(image).enhance(float(condition.split("_")[1]))
    return image.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", default="runs/robustness/summary.csv")
    args = parser.parse_args()

    data_file = Path(args.data).resolve()
    spec = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    root = Path(spec.get("path", data_file.parent))
    if not root.is_absolute():
        root = (data_file.parent / root).resolve()
    test_images = Path(spec["test"])
    if not test_images.is_absolute():
        test_images = root / test_images
    test_parts = list(test_images.parts)
    image_dir_indices = [index for index, part in enumerate(test_parts) if part == "images"]
    if not image_dir_indices:
        raise ValueError("The dataset test path must include an 'images' directory component.")
    test_parts[image_dir_indices[-1]] = "labels"
    test_labels = Path(*test_parts)
    image_files = sorted(p for p in test_images.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_files:
        raise ValueError(f"No test images found under {test_images}")

    model = YOLO(args.weights)
    rows = []
    with tempfile.TemporaryDirectory(prefix="mfmd_robustness_") as temporary:
        temporary = Path(temporary)
        for condition in CONDITIONS:
            images_out = temporary / condition / "images" / "test"
            labels_out = temporary / condition / "labels" / "test"
            images_out.mkdir(parents=True)
            labels_out.mkdir(parents=True)
            for source in image_files:
                with Image.open(source) as image:
                    transform(image.convert("RGB"), condition).save(images_out / f"{source.stem}.png")
                label = test_labels / f"{source.stem}.txt"
                if not label.is_file():
                    raise FileNotFoundError(f"Missing label for test image {source.name}: {label}")
                shutil.copy2(label, labels_out / label.name)
            generated = {"path": str(temporary / condition), "train": "images/test", "val": "images/test", "test": "images/test", "names": spec["names"]}
            generated_yaml = temporary / condition / "data.yaml"
            generated_yaml.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
            metrics = validate_paper(
                model, data=str(generated_yaml), split="test", imgsz=640, batch=args.batch,
                device=args.device, conf=0.25, iou=0.50,
            )
            fixed = fixed_threshold_metrics(metrics)
            rows.append({
                "condition": condition,
                "images": len(image_files),
                "confidence_threshold": fixed["confidence_threshold"],
                "matching_iou_threshold": fixed["matching_iou_threshold"],
                "tp": fixed["tp"],
                "fp": fixed["fp"],
                "fn": fixed["fn"],
                "precision": fixed["precision"],
                "recall": fixed["recall"],
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
