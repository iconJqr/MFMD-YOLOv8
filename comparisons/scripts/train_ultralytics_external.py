"""Train a comparison detector from an explicitly supplied external checkout.

This script intentionally does not install or download third-party repositories.
The caller must supply a trusted local checkout and record its immutable commit.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True, help="Trusted upstream repository checkout")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Upstream model YAML or pretrained checkpoint identifier")
    parser.add_argument("--api", choices=("YOLO", "RTDETR", "YOLOv10"), default="YOLO")
    parser.add_argument("--optimizer", required=True, help="Optimizer used by the confirmed original experiment")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--project", default="runs/comparisons")
    parser.add_argument("--name", default=None)
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--cos-lr", action="store_true")
    return parser.parse_args()


def git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> None:
    args = parse_args()
    repository = args.repo.resolve()
    if not repository.is_dir():
        raise FileNotFoundError(repository)
    sys.path.insert(0, str(repository))
    package = importlib.import_module("ultralytics")
    version = getattr(package, "__version__", "unknown")
    api_class = getattr(package, args.api, None)
    if api_class is None:
        raise RuntimeError(f"{args.api} is not available in {repository} (ultralytics {version}).")

    print(f"external_repository={repository}")
    print(f"external_commit={git_commit(repository)}")
    print(f"ultralytics_version={version}")

    for seed in (0, 1, 42):
        model = api_class(args.model)
        overrides = {
            "data": str(args.data.resolve()),
            "imgsz": 640,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch": args.batch,
            "seed": seed,
            "deterministic": True,
            "optimizer": args.optimizer,
            "device": args.device,
            "workers": args.workers,
            "project": args.project,
            "name": args.name or f"{Path(args.model).stem}_seed{seed}",
            "amp": True,
            "cos_lr": args.cos_lr,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "close_mosaic": 0,
        }
        if args.lr0 is not None:
            overrides["lr0"] = args.lr0
        if args.weight_decay is not None:
            overrides["weight_decay"] = args.weight_decay
        if args.momentum is not None:
            overrides["momentum"] = args.momentum
        model.train(**overrides)


if __name__ == "__main__":
    main()
