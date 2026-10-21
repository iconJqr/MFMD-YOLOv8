"""Run Table 9 one-factor-at-a-time sensitivity experiments over paper seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiment", choices=("all", "bra_topk_2", "bra_topk_8", "wiou_gamma_1_5",
                                                   "wiou_gamma_2_3", "wiou_delta_2", "wiou_delta_4"), default="all")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/sensitivity")
    args = parser.parse_args()

    settings = yaml.safe_load((ROOT / "configs" / "hyperparameter_sensitivity.yaml").read_text(encoding="utf-8"))
    protocol = yaml.safe_load((ROOT / "configs" / "train_paper.yaml").read_text(encoding="utf-8"))
    runs = settings["runs"] if args.experiment == "all" else {args.experiment: settings["runs"][args.experiment]}
    train_script = Path(__file__).with_name("train.py")
    for name, override in runs.items():
        for seed in protocol["seeds"]:
            command = [
                sys.executable, str(train_script), "--data", str(args.data),
                "--model", "mfmd_yolov8.yaml", "--iou-loss", "wiou_v3",
                "--weights", args.weights, "--device", args.device, "--workers", str(args.workers),
                "--project", args.project, "--name", f"{name}_seed{seed}", "--seed", str(seed),
                "--bra-topk", str(override["bra_topk"]),
                "--wiou-gamma", str(override["wiou_gamma"]),
                "--wiou-delta", str(override["wiou_delta"]),
            ]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
