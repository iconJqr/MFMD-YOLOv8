# MFMD-YOLOv8

Reference implementation for **MFMD-YOLOv8: An Enhanced Lightweight Framework for Mongolian Furniture Motif Detection**.

This reconstruction-oriented release contains the model definitions, RCS-OSA, BRA, the explicit CIoU/WIoU v3 loss switch, paper training settings, comparison-model documentation, dataset preparation templates, evaluation and visualization utilities, interpretability scripts, automated tests, and reproducibility notes. It does not contain the furniture photographs, annotations, checkpoints, experiment outputs, caches, or machine-specific paths.

Materials are labelled by provenance. `manuscript_reported` means a direct paper transcription, `reconstructed_from_manuscript` means a new implementation of a described procedure, `measured_run_pending` means that new experiments are required, and `synthetic_smoke_test_only` is never scientific evidence. See `docs/PROVENANCE_MATRIX.md`.

## Method and experiment scope

The released detector is based on YOLOv8n and uses three motif classes in this order: `geometry`, `animal`, `plant`. MFMD-YOLOv8 replaces C2f blocks in the backbone and neck with RCS-OSA, inserts BRA immediately after SPPF, and selects WIoU v3 for the bounding-box regression term. BCE classification loss and DFL are retained. The baseline and ablations select CIoU or WIoU v3 explicitly through `iou_loss`.

The proposed MFMD-YOLOv8 model contains 4.342 M parameters and requires 20.911 GFLOPs at an input resolution of 640×640 (exact parameter count: 4,342,313). Complexity is measured using standard `model.info(imgsz=640)` (retaining full GFLOPs without artificial division by 2 for GMACs).

The manuscript reports 640×640 inputs, batch size 8, SGD with initial learning rate 0.01, momentum 0.937, weight decay 0.0005, three warm-up epochs, cosine scheduling, AMP, up to 300 epochs, early-stopping patience 50, and seeds 0, 1, and 42. Training uses a fixed source-group partition and a fixed offline-augmented training set; online augmentation is disabled in the paper configuration.

## Installation

Use Python 3.10 with the PyTorch build appropriate for your CUDA installation:

```bash
pip install -r requirements.txt
pip install -e .
```

The code is based on the Ultralytics YOLOv8 source included in this directory. The upstream AGPL-3.0 license is retained in `LICENSE`.

## Dataset YAML

The image dataset is not distributed. Prepare a local YOLO-format dataset and copy `configs/data.example.yaml` to a local YAML, replacing `/path/to/mfmd_dataset`. The expected structure is:

```text
dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Labels use normalized YOLO `class x_center y_center width height` rows. The class indices must follow the order in the example YAML.

## Training

COCO-pretrained YOLOv8n weights are expected to be obtained separately. They are not included in this release.

One MFMD-YOLOv8 run:

```bash
python scripts/train.py --data /path/to/data.yaml --model mfmd_yolov8.yaml --iou-loss wiou_v3 --weights yolov8n.pt --seed 0
```

The `--model` argument resolves files under `configs/models`. To reproduce the three seeds sequentially:

```bash
python scripts/train_three_seeds.py --data /path/to/data.yaml --model mfmd_yolov8.yaml --iou-loss wiou_v3 --weights yolov8n.pt
```

For the baseline, use `yolov8n_baseline.yaml` with `--iou-loss ciou`. The eight combinations in the ablation table are listed in `configs/ablation.yaml` and can be launched with:

```bash
python scripts/train_ablation.py --data /path/to/data.yaml --experiment all --weights yolov8n.pt
```

The training script resets the WIoU v3 running mean for every independent run. This is important when multiple seeds or configurations are launched by one process.

## Comparison models

The 12 comparison detectors—Faster R-CNN, SSD-MobileNetV2, YOLOv5n, YOLOv8n, YOLOv8s, YOLOv9t, YOLOv10n, YOLOv11n, YOLOv12n, RTMDet-tiny, RT-DETR-L, and RT-DETR-X—are documented under `comparisons/`.

The package includes their registry, shared protocol, paper-reported results, a three-seed aggregation utility, and an external-Ultralytics launcher. Complete third-party repositories and pretrained weights are not redistributed. Entries whose exact source commit or optimizer cannot be established from the manuscript or retained logs are marked `pending_author_confirmation` rather than being replaced by an assumed implementation.

```bash
python comparisons/scripts/validate_registry.py
python comparisons/scripts/aggregate_three_seeds.py --input measured_results.csv --output comparison_summary.csv
```

## Evaluation and inference

```bash
python scripts/val.py --weights /path/to/best.pt --data /path/to/data.yaml --split test
python scripts/predict.py --weights /path/to/best.pt --source /path/to/images
python scripts/model_info.py  # prints model complexity at 640x640: 4.342 M parameters, 20.911 GFLOPs
python scripts/robustness_eval.py --weights /path/to/best.pt --data /path/to/data.yaml
python scripts/full_image_inference.py --weights /path/to/best.pt --source /path/to/held_out_photos --strategy combined
python evaluation/benchmark_model.py --weights /path/to/best.pt --data /path/to/data.yaml --output benchmark.json
```

The full-image script implements the manuscript's 640×640 windows, 512-pixel stride, 0.45 class-wise NMS, and optional overview pass. `robustness_eval.py` applies the four reported perturbations to the test images and keeps labels unchanged.

Training figures must be generated from retained `results.csv` files rather than reconstructed values:

```bash
python visualization/plot_training_curves.py --run YOLOv8n=/path/to/baseline/results.csv --run MFMD-YOLOv8=/path/to/mfmd/results.csv --output figures/training
```

## Dataset and installation smoke test

`dataset_tools/` contains ROI, VOC-to-YOLO, annotation validation, group split, leakage-check, and audited offline-augmentation utilities. Because the original image-level split is missing, `split_by_group.py` produces a new deterministic reconstruction and does not claim to restore the historical split.

The synthetic example tests software only:

```bash
python examples/smoke_test.py
python -m pytest -p no:cacheprovider tests -q
```

Before publication, build a clean ZIP that excludes caches, checkpoints, datasets, and generated runs:

```bash
python scripts/audit_release.py --allow-generated
python scripts/create_hash_manifest.py
python scripts/build_release_archive.py
```

## Reproducibility notes

See `docs/METHOD_CODE_MAP.md` for the paper-to-code map, `docs/REPRODUCIBILITY.md` for the protocol, `docs/PROVENANCE_MATRIX.md` for evidence status, `docs/AUTHOR_CONFIRMATION_CHECKLIST.md` for unresolved facts, `comparisons/README.md` for comparison-model status, and `docs/RELEASE_CHECKLIST.md` for the contents and exclusions of this package. The paper's private furniture dataset must be supplied locally under the stated copyright and access conditions.
