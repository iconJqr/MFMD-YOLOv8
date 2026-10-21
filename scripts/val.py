"""Evaluate a selected checkpoint without changing model weights."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from evaluation.paper_validation import fixed_threshold_metrics, validate_paper

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("runs/paper_eval.json"))
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="paper_test")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save confidence-scored YOLO txt predictions for error analysis")
    args = parser.parse_args()
    model = YOLO(args.weights)
    metrics = validate_paper(
        model, data=args.data, split=args.split, imgsz=640, batch=args.batch,
        device=args.device, conf=0.25, iou=0.50, project=args.project, name=args.name,
        save_txt=args.save_predictions, save_conf=args.save_predictions,
    )
    class_ap = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        class_name = model.names[int(class_id)]
        class_ap[class_name] = {
            "ap50": float(metrics.box.ap50[index]),
            "ap50_95": float(metrics.box.ap[index]),
        }
    payload = {
        "provenance": "measured_by_scripts_val",
        "split": args.split,
        "imgsz": 640,
        "batch": args.batch,
        "candidate_confidence_threshold": 0.25,
        "nms_iou_threshold": 0.50,
        "reported_metrics": {
            **fixed_threshold_metrics(metrics),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "ap_by_class": class_ap,
        },
        "prediction_directory": str(metrics.save_dir / "labels") if args.save_predictions else None,
        "ultralytics_best_f1_curve_precision": float(metrics.box.mp),
        "ultralytics_best_f1_curve_recall": float(metrics.box.mr),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
