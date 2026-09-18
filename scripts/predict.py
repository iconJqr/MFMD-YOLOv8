"""Run ordinary ROI prediction."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="mfmd")
    args = parser.parse_args()
    YOLO(args.weights).predict(
        source=args.source, imgsz=640, conf=args.conf, device=args.device,
        save=True, project=args.project, name=args.name)


if __name__ == "__main__":
    main()
