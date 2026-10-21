# MFMD-YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-AGPL%203.0-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-iconJqr%2FMFMD--YOLOv8-blue?logo=github)](https://github.com/iconJqr/MFMD-YOLOv8)

Reference implementation for **MFMD-YOLOv8: An Enhanced Lightweight Framework for Mongolian Furniture Motif Detection**.

This reconstruction-oriented release contains the model definitions, RCS-OSA, BRA, the explicit CIoU/WIoU v3 loss switch, paper training settings, comparison-model documentation, dataset preparation templates, evaluation and visualization utilities, interpretability scripts, automated tests, and reproducibility notes. It does not contain the furniture photographs, annotations, checkpoints, experiment outputs, caches, or machine-specific paths.

## Method and experiment scope

The released detector is based on YOLOv8n and uses three motif classes in this order: `geometry`, `animal`, `plant`. MFMD-YOLOv8 replaces C2f blocks in the backbone and neck with RCS-OSA, inserts BRA immediately after SPPF, and selects WIoU v3 for the bounding-box regression term. BCE classification loss and DFL are retained. The baseline and ablations select CIoU or WIoU v3 explicitly through `iou_loss`.

The proposed MFMD-YOLOv8 model contains 4.342 M parameters and requires 20.911 GFLOPs at an input resolution of 640x640 (exact parameter count: 4,342,313). The paper configuration records 640x640 inputs, batch size 8, SGD, cosine scheduling, three warm-up epochs, AMP, up to 300 epochs, early-stopping patience 50, and seeds 0, 1, and 42. Online geometric, colour, Mosaic, and MixUp transforms are disabled.

## Key Results

| Metric | MFMD-YOLOv8 | YOLOv8n (baseline) | Improvement |
|---|---:|---:|---:|
| mAP@0.5 | **87.9%** | 83.6% | +4.3 pp |
| mAP@0.5:0.95 | **59.3%** | 55.2% | +4.1 pp |
| Recall | **88.6%** | 77.1% | +11.5 pp |
| Precision | **86.7%** | 80.6% | +6.1 pp |
| Parameters | 4.3 M | 3.0 M | — |
| GFLOPs (640x640) | 20.9 | 8.2 | — |
| Speed (FPS) | 88.4 | 105.4 | — |

Results are mean values across three independent runs (seeds 0, 1, 42) on the fixed internal test set.

## Installation and dataset

Use Python 3.10 with the PyTorch build appropriate for your CUDA installation:

```bash
pip install -r requirements.txt
pip install -e .
```

The image dataset is not distributed. Prepare a local YOLO-format dataset and copy `configs/data.example.yaml` to a local YAML, replacing the placeholder path:

```text
dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

## Training

COCO-pretrained YOLOv8n weights are expected to be obtained separately. They are not included in this release.

```bash
python scripts/train.py --data /path/to/data.yaml --model mfmd_yolov8.yaml --iou-loss wiou_v3 --weights yolov8n.pt --seed 0
python scripts/train_three_seeds.py --data /path/to/data.yaml --model mfmd_yolov8.yaml --iou-loss wiou_v3 --weights yolov8n.pt
```

For the baseline, use `yolov8n_baseline.yaml` with `--iou-loss ciou`. Ablation and sensitivity configurations are recorded in `configs/ablation.yaml` and `configs/hyperparameter_sensitivity.yaml`.

```bash
python scripts/train_ablation.py --data /path/to/data.yaml --experiment all --weights yolov8n.pt
python scripts/train_sensitivity.py --data /path/to/data.yaml
```

## Comparison models

The 12 comparison detectors are documented under `comparisons/`. Complete third-party repositories and pretrained weights are not redistributed.

```bash
python comparisons/scripts/validate_registry.py
python comparisons/scripts/aggregate_three_seeds.py --input measured_results.csv --output comparison_summary.csv
```

## Evaluation and inference

```bash
python scripts/val.py --weights /path/to/best.pt --data /path/to/data.yaml --split test
python scripts/predict.py --weights /path/to/best.pt --source /path/to/images
python scripts/model_info.py
python scripts/robustness_eval.py --weights /path/to/best.pt --data /path/to/data.yaml
python scripts/full_image_inference.py --weights /path/to/best.pt --source /path/to/held_out_photos --strategy combined
python evaluation/benchmark_model.py --weights /path/to/best.pt --data /path/to/data.yaml --output benchmark.json
```

Use `evaluation/analyze_predictions.py` on saved YOLO-format predictions to compute fixed-threshold TP/FP/FN, true-positive IoU bins, and AP metrics.

## Verification and reproducibility

```bash
python examples/smoke_test.py
python -m pytest -p no:cacheprovider tests -q
```

Use `configs/train_paper.yaml`, `configs/offline_augmentation_reconstructed.yaml`, and `comparisons/shared_protocol.yaml` for the reported training, augmentation, and evaluation settings. Before publication, build a clean archive with:

```bash
python scripts/audit_release.py --allow-generated
python scripts/create_hash_manifest.py
python scripts/build_release_archive.py
```

## Citation

```bibtex
@article{ding2026mfmd,
  title={MFMD-YOLOv8: An Enhanced Lightweight Framework for Mongolian Furniture Motif Detection},
  author={Ding, Anning and Qiu, Guohua and Li, Qing and Kang, Yongkang and He, Chao and Jiang, Qi and Duo, Huaqiong},
  year={2026}
}
```

## License and acknowledgements

AGPL-3.0 (retained from Ultralytics YOLOv8). See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

This work was supported by the Key Research and Development and Achievement Transformation Program of Inner Mongolia Autonomous Region, China (Grant No. 2023YFHH0022 and Grant No. 2022YFDZ0031).
