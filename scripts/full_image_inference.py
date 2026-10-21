"""Full-image, sliding-window, or combined inference used for independent photographs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

def starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    points = list(range(0, length - window + 1, stride))
    if points[-1] != length - window:
        points.append(length - window)
    return points


def infer(model: YOLO, image: np.ndarray, strategy: str, conf: float, device: str):
    height, width = image.shape[:2]
    boxes, scores, classes = [], [], []
    if strategy in {"sliding", "combined"}:
        for top in starts(height, 640, 512):
            for left in starts(width, 640, 512):
                tile = image[top:min(top + 640, height), left:min(left + 640, width)]
                result = model.predict(tile, imgsz=640, rect=False, conf=conf, iou=0.50,
                                        device=device, verbose=False)[0]
                if len(result.boxes):
                    xyxy = result.boxes.xyxy.cpu()
                    xyxy[:, [0, 2]] += left
                    xyxy[:, [1, 3]] += top
                    boxes.append(xyxy)
                    scores.append(result.boxes.conf.cpu())
                    classes.append(result.boxes.cls.cpu())
    if strategy in {"full", "combined"}:
        result = model.predict(image, imgsz=640, rect=False, conf=conf, iou=0.50,
                               device=device, verbose=False)[0]
        if len(result.boxes):
            boxes.append(result.boxes.xyxy.cpu())
            scores.append(result.boxes.conf.cpu())
            classes.append(result.boxes.cls.cpu())
    if not boxes:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0)
    boxes, scores, classes = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
    keep_all = []
    for cls in classes.unique():
        idx = torch.where(classes == cls)[0]
        keep_all.append(idx[nms(boxes[idx], scores[idx], 0.45)])
    keep = torch.cat(keep_all)
    return boxes[keep], scores[keep], classes[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--strategy", choices=("full", "sliding", "combined"), default="combined")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", default="runs/full_image")
    args = parser.parse_args()

    model = YOLO(args.weights)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    paths = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        boxes, scores, classes = infer(model, image, args.strategy, args.conf, args.device)
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"{model.names[int(cls)]} {float(score):.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (46, 139, 87), 2)
            cv2.putText(image, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (46, 139, 87), 2)
        cv2.imwrite(str(output / path.name), image)


if __name__ == "__main__":
    main()
