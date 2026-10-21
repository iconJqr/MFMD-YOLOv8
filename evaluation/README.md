# Evaluation utilities

- `paper_validation.py` encodes manuscript validation thresholds: confidence
  0.25, ROI NMS IoU 0.50, plus fixed-threshold micro/per-class precision and
  recall from a confusion matrix.
- `select_best_checkpoint.py` applies the paper fitness
  `0.1*mAP50 + 0.9*mAP50_95` to a real validation `results.csv`; the internal
  test set is never used for selection.
- `benchmark_model.py` measures ROI forward-plus-GPU-NMS throughput with batch
  1 and 50 warm-up iterations, excluding file decoding and disk I/O.
- `full_image_evaluate.py` exports predictions and per-photo end-to-end time
  for full, sliding, or combined inference. Timing includes inference
  preprocessing, transfer, coordinate remapping, and post-processing, while
  excluding disk I/O. One untimed warm-up inference is used by default and can
  be changed with `--warmup-images`. Pass `--labels` to score the exported results.
- `analyze_predictions.py` computes fixed-threshold TP/FP/FN, precision/recall,
  per-class AP, mean IoU of true positives, and IoU bins from saved YOLO text
  predictions and labels. Use `--nms-iou 0.45` for full-image predictions
  (their final class-wise NMS uses 0.45); ROI NMS uses 0.50.

All generated values are new measurements. The scripts do not contain or claim
to regenerate the original run outputs.
