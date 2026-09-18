# MFMD-YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-AGPL%203.0-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-iconJqr%2FMFMD--YOLOv8-blue?logo=github)](https://github.com/iconJqr/MFMD-YOLOv8)

**MFMD-YOLOv8: An Enhanced Lightweight Framework for Mongolian Furniture Motif Detection**

Official reference implementation for reproducing the MFMD-YOLOv8 method.

A lightweight detector for animal, plant, and geometric motifs on traditional Mongolian painted furniture. Built on YOLOv8n with three key modifications:

- **RCS-OSA** — multi-scale feature aggregation and reuse
- **Bi-Level Routing Attention (BRA)** — content-dependent spatial interaction
- **Wise-IoU v3 (WIoU v3)** — dynamic non-monotonic bounding-box regression

## Key Results

| Metric              | MFMD-YOLOv8     | YOLOv8n (baseline) | Improvement |
|---------------------|-----------------|--------------------|-------------|
| mAP@0.5             | **87.9%**       | 83.6%              | +4.3 pp     |
| mAP@0.5:0.95        | **59.3%**       | 55.2%              | +4.1 pp     |
| Recall              | **88.6%**       | 77.1%              | +11.5 pp    |
| Precision           | **86.7%**       | 80.6%              | +6.1 pp     |
| Parameters          | 4.3 M           | 3.0 M              | —           |
| GFLOPs (640×640)    | 20.9            | 8.2                | —           |
| Speed (FPS)         | 88.4            | 105.4              | —           |

Results are mean values across three independent runs (seeds 0, 1, 42) on the fixed internal test set.

## Test Configuration

- OS: Windows 10 Pro
- CPU: Intel Core i5-13600KF
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 4060 Ti (8 GB VRAM)
- Software: Python 3.10.19, PyTorch 2.5.1, CUDA 12.1, cuDNN 9.1, Ultralytics 8.0.166

## Dataset

- 1,320 ROI images and 2,992 annotated instances from 237 traditional Mongolian furniture items
- Three classes (in order): `geometry`, `animal`, `plant`
- Source-group-aware partitioning (all observations from the same furniture item stay in one subset)
- Offline augmentation applied only to the training set; online augmentation disabled in the paper protocol

The image dataset itself is not distributed with this repository. Prepare a local YOLO-format dataset and copy `configs/data.example.yaml` to a local YAML, replacing the placeholder path. Expected structure:

```text
dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

## Method Overview

1. **RCS-OSA** replaces C2f blocks in the backbone and neck to support feature reuse across motifs of varying scales and fine brushstroke details.
2. **BRA** is inserted immediately after SPPF to perform content-dependent sparse attention, reducing interference from surface cracks, pigment loss, and dense ornamentation.
3. **WIoU v3** provides non-monotonic weighting of regression errors according to relative prediction quality.

The model contains 4,342,313 parameters and requires 20.911 GFLOPs at 640×640 input.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Training

COCO-pretrained YOLOv8n weights must be obtained separately.

```bash
# Single seed
python scripts/train.py --data /path/to/data.yaml --model mfmd_yolov8.yaml --iou-loss wiou_v3 --weights yolov8n.pt --seed 0

# Three seeds
python scripts/train_three_seeds.py --data /path/to/data.yaml --model mfmd_yolov8.yaml --iou-loss wiou_v3 --weights yolov8n.pt
```

Baseline: use `yolov8n_baseline.yaml` with `--iou-loss ciou`.

## Evaluation and Inference

```bash
python scripts/val.py --weights /path/to/best.pt --data /path/to/data.yaml --split test
python scripts/predict.py --weights /path/to/best.pt --source /path/to/images
python scripts/model_info.py
python scripts/full_image_inference.py --weights /path/to/best.pt --source /path/to/held_out_photos --strategy combined
```

## Documentation

- `docs/METHOD_CODE_MAP.md` — paper-to-code mapping
- `docs/REPRODUCIBILITY.md` — experimental protocol
- `docs/PROVENANCE_MATRIX.md` — evidence status
- `comparisons/` — comparison models and protocols

## Citation

```bibtex
@article{ding2026mfmd,
  title={MFMD-YOLOv8: An Enhanced Lightweight Framework for Mongolian Furniture Motif Detection},
  author={Ding, Anning and Qiu, Guohua and Li, Qing and Kang, Yongkang and He, Chao and Jiang, Qi and Duo, Huaqiong},
  year={2026}
}
```

## License

AGPL-3.0 (retained from Ultralytics YOLOv8). See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Acknowledgements

This work was supported by the Key Research and Development and Achievement Transformation Program of Inner Mongolia Autonomous Region, China (Grant No. 2023YFHH0022 and Grant No. 2022YFDZ0031).
