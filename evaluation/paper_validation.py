"""Validation settings and fixed operating-point metrics reported in the paper."""

from __future__ import annotations

import numpy as np

from ultralytics.models.yolo.detect.val import DetectionValidator


class ManuscriptDetectionValidator(DetectionValidator):
    """Collect a 0.25-confidence, 0.50-IoU confusion matrix alongside AP."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.confusion_matrix.conf = 0.25
        self.confusion_matrix.iou_thres = 0.50

    def update_metrics(self, preds, batch):
        # The bundled validator only updates its confusion matrix when plots are
        # enabled. Toggle that branch for this call without generating figures.
        plots = self.args.plots
        self.args.plots = True
        try:
            super().update_metrics(preds, batch)
        finally:
            self.args.plots = plots


def validate_paper(model, **kwargs):
    """Run the manuscript's 640-pixel validation protocol."""
    kwargs.setdefault("imgsz", 640)
    kwargs.setdefault("conf", 0.25)
    kwargs.setdefault("iou", 0.50)
    kwargs.setdefault("plots", False)
    return model.val(validator=ManuscriptDetectionValidator, **kwargs)


def fixed_threshold_metrics(metrics) -> dict[str, float | int]:
    """Return micro precision/recall and TP/FP/FN from the fixed-threshold matrix."""
    confusion = getattr(metrics, "confusion_matrix", None)
    if confusion is None:
        raise ValueError("Validation did not return a confusion matrix.")
    matrix = np.asarray(confusion.matrix, dtype=np.int64)
    tp = int(np.diag(matrix)[:-1].sum())
    fp = int(matrix[:-1, :].sum() - tp)
    fn = int(matrix[:, :-1].sum() - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    per_class = {}
    for class_id in range(matrix.shape[0] - 1):
        class_tp = int(matrix[class_id, class_id])
        class_fp = int(matrix[class_id, :].sum() - class_tp)
        class_fn = int(matrix[:, class_id].sum() - class_tp)
        per_class[str(class_id)] = {
            "tp": class_tp,
            "fp": class_fp,
            "fn": class_fn,
            "precision": class_tp / (class_tp + class_fp) if class_tp + class_fp else 0.0,
            "recall": class_tp / (class_tp + class_fn) if class_tp + class_fn else 0.0,
        }
    return {
        "confidence_threshold": 0.25,
        "matching_iou_threshold": 0.50,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "per_class": per_class,
    }
