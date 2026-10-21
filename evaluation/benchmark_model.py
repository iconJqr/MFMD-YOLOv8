"""Measure checkpoint metrics and runtime; writes measured values with context."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.utils.ops import non_max_suppression
from evaluation.paper_validation import fixed_threshold_metrics, validate_paper


def roi_inputs(data_file: Path, split: str, device: torch.device) -> list[torch.Tensor]:
    spec = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    root = Path(spec.get("path", data_file.parent))
    if not root.is_absolute():
        root = (data_file.parent / root).resolve()
    image_dir = Path(spec[split])
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    paths = sorted(path for path in image_dir.iterdir()
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not paths:
        raise ValueError(f"No {split} ROI images found under {image_dir}")

    inputs = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable ROI image: {path}")
        if image.shape[:2] != (640, 640):
            raise ValueError(f"ROI timing requires 640x640 prepared images: {path} has {image.shape[:2]}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
        inputs.append(tensor.to(device=device, dtype=torch.float32).div_(255.0))
    return inputs


def measure_roi_fps(model: YOLO, data_file: Path, split: str, device_arg: str, warmup: int = 50) -> dict:
    if device_arg == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        if device_arg == "cuda":
            requested = f"cuda:{torch.cuda.current_device()}"
        elif device_arg.startswith("cuda:"):
            requested = device_arg
        else:
            requested = f"cuda:{device_arg.split(',')[0]}"
        device = torch.device(requested)
    inputs = roi_inputs(data_file, split, device)
    network = model.model.to(device).eval()
    parameters = sum(parameter.numel() for parameter in network.parameters())
    if hasattr(network, "fuse"):
        network.fuse(verbose=False)

    def forward_and_nms(tensor: torch.Tensor) -> None:
        output = network(tensor)
        predictions = output[0] if isinstance(output, tuple) else output
        non_max_suppression(
            predictions, conf_thres=0.25, iou_thres=0.50, multi_label=True,
            agnostic=False, max_det=300,
        )

    for index in range(warmup):
        forward_and_nms(inputs[index % len(inputs)])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for tensor in inputs:
        forward_and_nms(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "images": len(inputs),
        "warmup_iterations": warmup,
        "batch_size": 1,
        "confidence_threshold": 0.25,
        "nms_iou_threshold": 0.50,
        "nms_device": device.type,
        "total_seconds_forward_plus_gpu_nms": elapsed,
        "mean_seconds_per_image": elapsed / len(inputs),
        "fps_forward_plus_gpu_nms": len(inputs) / elapsed if elapsed else None,
        "parameters_before_inference_fusion": parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    args = parser.parse_args()

    model = YOLO(args.weights)
    result = validate_paper(
        model, data=args.data, split=args.split, imgsz=640, batch=args.batch,
        device=args.device, conf=0.25, iou=0.50, verbose=False,
    )
    metrics = getattr(result, "results_dict", {})
    fixed = fixed_threshold_metrics(result)
    module = model.model
    params = sum(parameter.numel() for parameter in module.parameters())
    fps = measure_roi_fps(model, Path(args.data).resolve(), args.split, args.device)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "measured_by_this_script",
        "checkpoint": str(Path(args.weights).resolve()),
        "dataset_yaml": str(Path(args.data).resolve()),
        "split": args.split,
        "imgsz": 640,
        "batch": args.batch,
        "metrics": {str(key): float(value) for key, value in metrics.items()},
        "reported_operating_point": fixed,
        "roi_throughput": fps,
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
        "timing_note": "ROI FPS includes forward propagation and NMS after 50 warm-up iterations; image decoding, host-to-device transfer, and disk I/O are excluded. Only GPU runs are comparable with the manuscript's reported hardware.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
