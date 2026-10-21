# Comparison-model reproducibility

This directory documents the 12 comparison detectors reported in the manuscript. It does **not** vendor their complete third-party repositories or checkpoints. Each detector should be trained in an isolated environment created from its cited upstream repository, while using the same fixed furniture-source split and evaluation protocol.

## Included files

- `model_registry.yaml`: model identity, upstream project, candidate entrypoint, paper-reported complexity, and verification status.
- `shared_protocol.yaml`: dataset, training-seed, checkpoint-selection, and evaluation constraints common to all models.
- `paper_reported_results.csv`: the values transcribed from the manuscript; these are reference values, not newly generated outputs.
- `scripts/train_ultralytics_external.py`: launcher for comparison models implemented by an external Ultralytics checkout.
- `scripts/aggregate_three_seeds.py`: computes mean and sample standard deviation from three per-seed result rows.
- `scripts/validate_registry.py`: checks that all 12 comparison models are represented and that source metadata is explicit.
- `THIRD_PARTY_CODE.md`: repository and licensing guidance.

## Important status distinction

`partially_verified` and `candidate_entrypoint` distinguish model details stated in the manuscript from a candidate public entrypoint. `source_details_not_recorded` means the repository, revision, or entrypoint is not identified in this release. Optimizer settings in the registry are transcribed from Table 4.

The registry separates two goals:

1. transparent disclosure of every detector reported in the paper; and
2. executable reproduction, using the source revision, configuration, and three seeds recorded for each new run.

## External Ultralytics models

Run the launcher with a clean upstream checkout. Do not point it at this MFMD fork unless reproducing YOLOv8n/MFMD ablations.

```bash
python comparisons/scripts/train_ultralytics_external.py \
  --repo /path/to/upstream/repository \
  --data /path/to/data.yaml \
  --model yolov8s.pt \
  --api YOLO \
  --optimizer SGD \
  --lr0 0.01 \
  --weight-decay 0.0005 \
  --momentum 0.937 \
  --project runs/comparisons/yolov8s
```

The launcher runs seeds 0, 1, and 42 sequentially and records the imported package version. It applies the manuscript's cosine schedule (`lrf=0.01`), three-epoch warm-up (`momentum=0.8`, `bias_lr=0.1`), AMP, 640-pixel inputs, batch size 8, 300-epoch maximum, 50-epoch patience, confidence threshold 0.25, ROI NMS IoU 0.50, and disabled online augmentation. Provide the model-specific optimizer, initial learning rate, and weight decay from Table 4. For YAML architecture definitions, also supply the official COCO checkpoint with `--weights`.

Faster R-CNN, SSD-MobileNetV2, and RTMDet-tiny require their own upstream training APIs. Their source details are not recorded in `model_registry.yaml`; use the corresponding upstream implementation and record its revision and configuration for each run.
