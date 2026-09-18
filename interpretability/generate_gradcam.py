"""Generate class-specific P3/P4/P5 and fused Grad-CAM maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.maximum(array, 0)
    return (array - array.min()) / (array.max() - array.min() + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="YOLO labels; when supplied, generate only classes present per image")
    parser.add_argument("--architecture", choices=("yolov8n_baseline", "mfmd_yolov8"), required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/gradcam"))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    configuration = yaml.safe_load((ROOT / "interpretability" / "select_layers.yaml").read_text(encoding="utf-8"))
    layer_ids = configuration[args.architecture]
    model = YOLO(args.weights).model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model.to(device).eval()
    paths = [args.source] if args.source.is_file() else sorted(
        path for path in args.source.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    args.output.mkdir(parents=True, exist_ok=True)

    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        resized = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].to(device) / 255.0
        classes = set(range(model.model[-1].nc))
        if args.labels:
            label_file = args.labels / f"{path.stem}.txt"
            classes = {int(line.split()[0]) for line in label_file.read_text(encoding="utf-8").splitlines()} if label_file.exists() else set()
        for cls in sorted(classes):
            activations, gradients, handles = {}, {}, []
            for level, index in layer_ids.items():
                def capture(_module, _inputs, output, name=level):
                    activations[name] = output
                    output.register_hook(lambda gradient, key=name: gradients.__setitem__(key, gradient))
                handles.append(model.model[index].register_forward_hook(capture))
            model.zero_grad(set_to_none=True)
            prediction = model(tensor)
            decoded = prediction[0] if isinstance(prediction, tuple) else prediction
            decoded[:, 4 + cls].max().backward()
            maps = []
            for level in layer_ids:
                activation, gradient = activations[level], gradients[level]
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * activation).sum(1, keepdim=True))
                cam = F.interpolate(cam, (640, 640), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
                cam = normalize(cam)
                np.save(args.output / f"{path.stem}_class{cls}_{level}.npy", cam)
                maps.append(cam)
            fused = normalize(np.mean(maps, axis=0))
            np.save(args.output / f"{path.stem}_class{cls}_fusion.npy", fused)
            heat = cv2.applyColorMap(np.uint8(fused * 255), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(resized, 0.55, heat, 0.45, 0)
            cv2.imwrite(str(args.output / f"{path.stem}_class{cls}_fusion.png"), overlay)
            for handle in handles:
                handle.remove()


if __name__ == "__main__":
    main()
