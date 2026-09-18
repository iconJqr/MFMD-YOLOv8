"""Build the paper models and report parameters and FLOPs at 640x640 input resolution.

Expected MFMD-YOLOv8 complexity:
The proposed MFMD-YOLOv8 model contains 4.342 M parameters and requires 20.911 GFLOPs
at an input resolution of 640x640 (exact parameter count: 4,342,313, standard model.info(imgsz=640)).
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


for name in ("yolov8n_baseline.yaml", "mfmd_yolov8.yaml"):
    print(f"\n=== {name} ===")
    YOLO(str(ROOT / "configs" / "models" / name), task="detect").info(detailed=False, verbose=True, imgsz=640)
