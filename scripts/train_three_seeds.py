"""Run the manuscript's three independent seeds sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="mfmd_yolov8.yaml")
    parser.add_argument("--iou-loss", choices=("ciou", "wiou_v3"), default="wiou_v3")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/paper")
    args = parser.parse_args()

    script = Path(__file__).with_name("train.py")
    for seed in (0, 1, 42):
        command = [
            sys.executable, str(script), "--data", args.data, "--model", args.model,
            "--iou-loss", args.iou_loss, "--weights", args.weights, "--device", args.device,
            "--workers", str(args.workers), "--project", args.project, "--seed", str(seed),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
