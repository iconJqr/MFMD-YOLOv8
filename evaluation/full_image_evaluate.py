"""Evaluate full/sliding/combined inference on held-out photographs.

For standard resized-image evaluation use ``scripts/val.py``. This command
records paper-protocol predictions, error metrics, and timing when annotations
are supplied, without hiding tiling or NMS behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultralytics import YOLO
from full_image_inference import infer
from evaluation.analyze_predictions import analyze


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--strategy", choices=("full", "sliding", "combined"), default="combined")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup-images", type=int, default=1,
                        help="Untimed setup/warm-up inferences before collecting per-photo timing")
    parser.add_argument("--labels", type=Path,
                        help="Optional YOLO label directory; when supplied, write paper metrics and error counts")
    parser.add_argument("--output", type=Path, default=Path("runs/full_image_predictions"))
    args = parser.parse_args()
    model = YOLO(args.weights)
    label_dir = args.output / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    elapsed_times = []
    image_paths = sorted(item for item in args.images.iterdir() if item.suffix.lower() in EXTENSIONS)
    if args.warmup_images < 0:
        raise ValueError("--warmup-images must be zero or greater")
    if not image_paths:
        raise ValueError(f"No supported images found in {args.images}")
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise ValueError(f"Unreadable warm-up image: {image_paths[0]}")
    for _ in range(args.warmup_images):
        infer(model, first_image, args.strategy, args.conf, args.device)
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            manifest.append({"image": path.name, "status": "unreadable", "detections": 0,
                             "processing_seconds_excluding_disk_io": ""})
            continue
        height, width = image.shape[:2]
        if torch.cuda.is_available() and args.device != "cpu":
            device_text = args.device.removeprefix("cuda:").split(",")[0]
            device_id = int(device_text) if device_text.isdigit() else torch.cuda.current_device()
            torch.cuda.synchronize(device_id)
        start = time.perf_counter()
        boxes, scores, classes = infer(model, image, args.strategy, args.conf, args.device)
        if torch.cuda.is_available() and args.device != "cpu":
            torch.cuda.synchronize(device_id)
        elapsed = time.perf_counter() - start
        elapsed_times.append(elapsed)
        rows = []
        for box, score, cls in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
            x1, y1, x2, y2 = box
            rows.append(f"{int(cls)} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} "
                        f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f} {score:.8f}")
        (label_dir / f"{path.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        manifest.append({"image": path.name, "status": "complete", "detections": len(rows),
                         "processing_seconds_excluding_disk_io": elapsed})
    with (args.output / "prediction_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "status", "detections",
                                                    "processing_seconds_excluding_disk_io"))
        writer.writeheader()
        writer.writerows(manifest)
    mean_seconds = sum(elapsed_times) / len(elapsed_times) if elapsed_times else None
    (args.output / "TIMING.json").write_text(
        json.dumps({
            "provenance": "measured_by_full_image_evaluate",
            "strategy": args.strategy,
            "images": len(elapsed_times),
            "warmup_images": args.warmup_images,
            "mean_seconds_per_image_excluding_disk_io": mean_seconds,
            "confidence_threshold": args.conf,
            "tile_nms_iou_threshold": 0.50,
            "global_classwise_nms_iou_threshold": 0.45,
            "device": args.device,
            "batch_size": 1,
        }, indent=2) + "\n", encoding="utf-8")
    if args.labels:
        metrics = analyze(args.labels, label_dir, confidence_threshold=args.conf, nms_iou_threshold=0.45)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "PROVENANCE.txt").write_text(
        "Generated predictions and timings; not a manuscript result. Score the saved detections "
        "against the retained full-image annotations using the fixed paper evaluator.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
