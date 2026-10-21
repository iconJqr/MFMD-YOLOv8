"""Compute fixed-threshold errors, IoU bins, and AP from YOLO text files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics.utils.metrics import ap_per_class


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
CLASS_NAMES = {0: "geometry", 1: "animal", 2: "plant"}


def read_yolo_file(path: Path, with_confidence: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        required = 6 if with_confidence else 5
        if len(values) < required:
            raise ValueError(f"Expected {required} columns in {path}: {line}")
        rows.append(values[:required])
    if not rows:
        return np.empty((0, 4)), np.empty(0, dtype=np.int64), np.empty(0) if with_confidence else None
    array = np.asarray(rows, dtype=np.float64)
    classes = array[:, 0].astype(np.int64)
    xc, yc, width, height = array[:, 1:5].T
    boxes = np.stack((xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2), axis=1)
    scores = array[:, 5] if with_confidence else None
    return boxes, classes, scores


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if not len(boxes1) or not len(boxes2):
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = np.maximum(rb - lt, 0).prod(axis=2)
    area1 = np.maximum(boxes1[:, 2:] - boxes1[:, :2], 0).prod(axis=1)
    area2 = np.maximum(boxes2[:, 2:] - boxes2[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area1[:, None] + area2[None, :] - intersection, 1e-12)


def match_at_threshold(pred_boxes: np.ndarray, pred_classes: np.ndarray, gt_boxes: np.ndarray,
                       gt_classes: np.ndarray, threshold: float) -> tuple[np.ndarray, list[float]]:
    matched = np.zeros(len(pred_boxes), dtype=bool)
    matched_ious: list[float] = []
    ious = box_iou(pred_boxes, gt_boxes)
    candidates = np.argwhere((ious >= threshold) & (pred_classes[:, None] == gt_classes[None, :]))
    ranked = sorted(((float(ious[p, g]), int(p), int(g)) for p, g in candidates), reverse=True)
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    for overlap, pred_index, gt_index in ranked:
        if pred_index in used_predictions or gt_index in used_targets:
            continue
        used_predictions.add(pred_index)
        used_targets.add(gt_index)
        matched[pred_index] = True
        matched_ious.append(overlap)
    return matched, matched_ious


def analyze(labels_dir: Path, predictions_dir: Path, confidence_threshold: float = 0.25,
            nms_iou_threshold: float = 0.50) -> dict:
    label_paths = sorted(labels_dir.glob("*.txt"))
    if not label_paths:
        raise ValueError(f"No label files found in {labels_dir}")
    label_stems = {path.stem for path in label_paths}
    prediction_stems = {path.stem for path in predictions_dir.glob("*.txt")}
    if prediction_stems != label_stems:
        missing = sorted(label_stems - prediction_stems)
        extra = sorted(prediction_stems - label_stems)
        raise ValueError(f"Prediction/label file sets differ; missing={missing[:5]}, extra={extra[:5]}")

    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "iou_values": []})
    all_correct, all_confidence, all_prediction_classes, all_target_classes = [], [], [], []
    for label_path in label_paths:
        gt_boxes, gt_classes, _ = read_yolo_file(label_path, with_confidence=False)
        pred_boxes, pred_classes, scores = read_yolo_file(predictions_dir / label_path.name, with_confidence=True)
        keep = scores >= confidence_threshold
        pred_boxes, pred_classes, scores = pred_boxes[keep], pred_classes[keep], scores[keep]
        correct = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
        for column, threshold in enumerate(IOU_THRESHOLDS):
            correct[:, column], _ = match_at_threshold(pred_boxes, pred_classes, gt_boxes, gt_classes, float(threshold))
        all_correct.append(correct)
        all_confidence.append(scores)
        all_prediction_classes.append(pred_classes)
        all_target_classes.append(gt_classes)

        matched, matched_ious = match_at_threshold(pred_boxes, pred_classes, gt_boxes, gt_classes, 0.50)
        for class_id in set(gt_classes.tolist()) | set(pred_classes.tolist()):
            cls_predictions = pred_classes == class_id
            cls_targets = gt_classes == class_id
            cls_matched, cls_ious = match_at_threshold(
                pred_boxes[cls_predictions], pred_classes[cls_predictions],
                gt_boxes[cls_targets], gt_classes[cls_targets], 0.50,
            )
            record = per_class[int(class_id)]
            record["tp"] += int(cls_matched.sum())
            record["fp"] += int(len(cls_matched) - cls_matched.sum())
            record["fn"] += int(cls_targets.sum() - cls_matched.sum())
            record["iou_values"].extend(cls_ious)

    if not all_target_classes or sum(map(len, all_target_classes)) == 0:
        raise ValueError("The supplied labels contain no ground-truth instances.")
    correct = np.concatenate(all_correct, axis=0) if all_correct else np.empty((0, len(IOU_THRESHOLDS)), bool)
    confidences = np.concatenate(all_confidence) if all_confidence else np.empty(0)
    prediction_classes = np.concatenate(all_prediction_classes) if all_prediction_classes else np.empty(0, int)
    target_classes = np.concatenate(all_target_classes)
    _, _, _, _, _, ap, ap_classes = ap_per_class(correct, confidences, prediction_classes, target_classes)

    total_tp = sum(value["tp"] for value in per_class.values())
    total_fp = sum(value["fp"] for value in per_class.values())
    total_fn = sum(value["fn"] for value in per_class.values())
    all_ious = [iou for value in per_class.values() for iou in value["iou_values"]]
    iou_bins = {
        "0.50_to_0.70_percent": 100 * sum(0.50 <= value < 0.70 for value in all_ious) / max(len(all_ious), 1),
        "0.70_to_0.90_percent": 100 * sum(0.70 <= value < 0.90 for value in all_ious) / max(len(all_ious), 1),
        "0.90_to_1.00_percent": 100 * sum(0.90 <= value <= 1.00 for value in all_ious) / max(len(all_ious), 1),
    }
    class_ap = {CLASS_NAMES.get(int(class_id), str(int(class_id))): {
        "ap50": float(ap[index, 0]), "ap50_95": float(ap[index].mean())
    } for index, class_id in enumerate(ap_classes)}
    class_results = {}
    for class_id, record in sorted(per_class.items()):
        tp, fp, fn = record["tp"], record["fp"], record["fn"]
        class_results[CLASS_NAMES.get(class_id, str(class_id))] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "mean_iou_of_tp": float(np.mean(record["iou_values"])) if record["iou_values"] else 0.0,
        }
    macro_map50 = float(np.mean(ap[:, 0])) if len(ap) else 0.0
    macro_map = float(np.mean(ap.mean(axis=1))) if len(ap) else 0.0
    return {
        "provenance": "computed_from_saved_predictions_and_labels",
        "images": len(label_paths),
        "confidence_threshold": confidence_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "matching_iou_threshold": 0.50,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        "map50": macro_map50,
        "map50_95": macro_map,
        "ap_by_class": class_ap,
        "error_counts_by_class": class_results,
        "mean_iou_of_true_positives": float(np.mean(all_ious)) if all_ious else 0.0,
        "true_positive_iou_bins": iou_bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=Path("runs/detection_analysis.json"))
    args = parser.parse_args()
    result = analyze(args.labels, args.predictions, args.confidence, args.nms_iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
