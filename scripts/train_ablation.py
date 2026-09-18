"""Run one or all ablation configurations, each with seeds 0, 1, and 42."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--experiment", choices=("all", *(f"exp{i}" for i in range(1, 9))), default="all")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/ablation")
    args = parser.parse_args()

    experiments = yaml.safe_load((ROOT / "configs" / "ablation.yaml").read_text(encoding="utf-8"))
    selected = experiments if args.experiment == "all" else {args.experiment: experiments[args.experiment]}
    script = Path(__file__).with_name("train.py")
    for exp_name, exp in selected.items():
        for seed in (0, 1, 42):
            command = [
                sys.executable, str(script), "--data", args.data, "--model", exp["model"],
                "--iou-loss", exp["iou_loss"], "--weights", args.weights, "--device", args.device,
                "--workers", str(args.workers), "--project", args.project,
                "--name", f"{exp_name}_seed{seed}", "--seed", str(seed),
            ]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
