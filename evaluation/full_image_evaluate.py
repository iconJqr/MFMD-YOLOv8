"""Evaluate full/sliding/combined inference through a generated YOLO data YAML.

For standard resized-image evaluation use ``scripts/val.py``. This command
records full-image predictions as YOLO text so they can be scored by a chosen
external detection evaluator without hiding tiling/NMS behavior.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultralytics import YOLO
from full_image_inference import infer


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--strategy", choices=("full", "sliding", "combined"), default="combined")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=Path("runs/full_image_predictions"))
    args = parser.parse_args()
    model = YOLO(args.weights)
    label_dir = args.output / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(item for item in args.images.iterdir() if item.suffix.lower() in EXTENSIONS):
        image = cv2.imread(str(path))
        if image is None:
            manifest.append({"image": path.name, "status": "unreadable", "detections": 0})
            continue
        height, width = image.shape[:2]
        boxes, scores, classes = infer(model, image, args.strategy, args.conf, args.device)
        rows = []
        for box, score, cls in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
            x1, y1, x2, y2 = box
            rows.append(f"{int(cls)} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} "
                        f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f} {score:.8f}")
        (label_dir / f"{path.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        manifest.append({"image": path.name, "status": "complete", "detections": len(rows)})
    with (args.output / "prediction_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "status", "detections"))
        writer.writeheader()
        writer.writerows(manifest)
    (args.output / "PROVENANCE.txt").write_text(
        "Generated predictions only; not a manuscript result. Score against retained full-image annotations "
        "using one fixed evaluator and report its version.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
