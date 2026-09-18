"""Measure checkpoint metrics and runtime; writes measured values with context."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO, __version__ as ultralytics_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    args = parser.parse_args()

    model = YOLO(args.weights)
    result = model.val(data=args.data, split=args.split, imgsz=640, batch=args.batch, device=args.device, verbose=False)
    metrics = getattr(result, "results_dict", {})
    speed = getattr(result, "speed", {})
    inference_ms = float(speed.get("inference", float("nan")))
    module = model.model
    params = sum(parameter.numel() for parameter in module.parameters())
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "measured_by_this_script",
        "checkpoint": str(Path(args.weights).resolve()),
        "dataset_yaml": str(Path(args.data).resolve()),
        "split": args.split,
        "imgsz": 640,
        "batch": args.batch,
        "metrics": {str(key): float(value) for key, value in metrics.items()},
        "speed_ms_per_image": {str(key): float(value) for key, value in speed.items()},
        "fps_from_inference_only": 1000.0 / inference_ms if inference_ms > 0 else None,
        "parameters": params,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ultralytics_source_version": ultralytics_version,
        },
        "timing_note": "FPS is inference-only from validator timing; compare only runs with identical scope and hardware.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
