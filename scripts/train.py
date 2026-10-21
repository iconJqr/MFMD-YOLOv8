"""Train one paper configuration with an explicit CIoU/WIoU-v3 switch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.block import BiLevelRoutingAttention
from ultralytics.utils.metrics import WIoU_Scale


MODELS = ROOT / "configs" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="YOLO-format dataset YAML")
    parser.add_argument("--model", default="mfmd_yolov8.yaml", help="File in configs/models, or another YAML path")
    parser.add_argument("--iou-loss", choices=("ciou", "wiou_v3"), default="wiou_v3")
    parser.add_argument("--seed", type=int, choices=(0, 1, 42), default=0)
    parser.add_argument("--bra-topk", type=int, help="Override BRA routed-region count for sensitivity runs")
    parser.add_argument("--wiou-gamma", type=float, help="Override WIoU v3 gamma for sensitivity runs")
    parser.add_argument("--wiou-delta", type=float, help="Override WIoU v3 delta for sensitivity runs")
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
    paper_seeds = tuple(train_cfg.pop("seeds"))
    default_bra_topk = int(train_cfg.pop("bra_topk"))
    default_wiou_gamma = float(train_cfg.pop("wiou_gamma"))
    default_wiou_delta = float(train_cfg.pop("wiou_delta"))
    bra_topk = args.bra_topk if args.bra_topk is not None else default_bra_topk
    wiou_gamma = args.wiou_gamma if args.wiou_gamma is not None else default_wiou_gamma
    wiou_delta = args.wiou_delta if args.wiou_delta is not None else default_wiou_delta
    if args.seed not in paper_seeds:
        raise ValueError(f"seed={args.seed} is not one of the paper seeds {paper_seeds}.")
    if bra_topk < 1:
        raise ValueError("BRA top-k must be positive.")
    # Keep classification-only augmentation controls visible in the released
    # paper configuration, but do not forward unsupported keys to YOLO detect.
    detect_inapplicable = {
        "erasing": train_cfg.pop("erasing"),
        "auto_augment": train_cfg.pop("auto_augment"),
    }
    if detect_inapplicable != {"erasing": 0.0, "auto_augment": None}:
        raise ValueError("The paper protocol requires erasing=0.0 and auto_augment=None.")
    train_cfg.update(
        data=str(args.data),
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name or f"{model_path.stem}_{args.iou_loss}_seed{args.seed}",
        iou_loss=args.iou_loss,
        wiou_gamma=wiou_gamma,
        wiou_delta=wiou_delta,
    )

    model = YOLO(str(model_path), task="detect")
    bra_modules = [module for module in model.model.modules() if isinstance(module, BiLevelRoutingAttention)]
    if args.bra_topk is not None and not bra_modules:
        raise ValueError("--bra-topk was supplied, but the selected architecture has no BRA module.")
    for module in bra_modules:
        module.topk = bra_topk
        module.router.topk = bra_topk
    for section in ("backbone", "head"):
        for layer in model.model.yaml.get(section, []):
            if layer[2] == "BiLevelRoutingAttention":
                layer[3][8] = bra_topk
    WIoU_Scale.configure(wiou_gamma, wiou_delta)
    if args.weights.lower() != "none":
        model.load(args.weights)
    model.train(**train_cfg)
    run_dir = Path(model.trainer.save_dir)
    method_config = {
        "provenance": "manuscript_protocol_with_cli_overrides",
        "model_yaml": str(model_path.resolve()),
        "data_yaml": str(args.data.resolve()),
        "pretrained_weights": args.weights,
        "seed": args.seed,
        "paper_seeds": list(paper_seeds),
        "iou_loss": args.iou_loss,
        "bra_topk": bra_topk,
        "wiou_gamma": wiou_gamma,
        "wiou_delta": wiou_delta,
        "training_arguments": train_cfg,
    }
    (run_dir / "method_config.json").write_text(json.dumps(method_config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
