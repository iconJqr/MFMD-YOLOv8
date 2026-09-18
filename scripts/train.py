"""Train one paper configuration with an explicit CIoU/WIoU-v3 switch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


MODELS = ROOT / "configs" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="YOLO-format dataset YAML")
    parser.add_argument("--model", default="mfmd_yolov8.yaml", help="File in configs/models, or another YAML path")
    parser.add_argument("--iou-loss", choices=("ciou", "wiou_v3"), default="wiou_v3")
    parser.add_argument("--seed", type=int, choices=(0, 1, 42), default=0)
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="COCO checkpoint, or 'none'")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/paper")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        model_path = MODELS / args.model
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    train_cfg = yaml.safe_load((ROOT / "configs" / "train_paper.yaml").read_text(encoding="utf-8"))
    train_cfg.update(
        data=str(args.data),
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name or f"{model_path.stem}_{args.iou_loss}_seed{args.seed}",
        iou_loss=args.iou_loss,
    )

    model = YOLO(str(model_path), task="detect")
    if args.weights.lower() != "none":
        model.load(args.weights)
    model.train(**train_cfg)


if __name__ == "__main__":
    main()
