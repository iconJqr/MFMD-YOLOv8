"""CPU smoke test: create toy data and run one untrained forward pass."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import cv2
import torch
from generate_synthetic_dataset import generate
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()
    context = tempfile.TemporaryDirectory() if args.workdir is None else None
    workdir = Path(context.name) if context else args.workdir
    data = generate(workdir / "data", size=224)
    image = cv2.imread(str(next((workdir / "data" / "images" / "val").glob("*.png"))))
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    model = YOLO(str(ROOT / "configs" / "models" / "mfmd_yolov8.yaml"), task="detect").model.eval()
    with torch.no_grad():
        output = model(tensor)
    assert output is not None and data.is_file()
    print("MFMD-YOLOv8 synthetic CPU smoke test passed")
    if context:
        context.cleanup()


if __name__ == "__main__":
    main()
