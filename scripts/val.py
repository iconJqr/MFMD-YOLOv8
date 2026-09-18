"""Evaluate a selected checkpoint without changing model weights."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    YOLO(args.weights).val(data=args.data, split=args.split, imgsz=640, batch=args.batch, device=args.device)


if __name__ == "__main__":
    main()
