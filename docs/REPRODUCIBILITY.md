# Reproducibility protocol

1. Prepare the private dataset with source-group-disjoint `train`, `val`, and `test` splits. Keep all images and derived ROIs from one furniture item in one subset.
2. Use 640×640 RGB inputs and the three class indices from `configs/data.example.yaml`.
3. Apply the manuscript's offline training-only augmentation before training and do not apply online augmentation during the reported runs. The final offline set contained 1051 original plus 3153 augmented training images.
4. Start from a separately obtained COCO-pretrained YOLOv8n checkpoint. Run `scripts/train_three_seeds.py` with seeds 0, 1, and 42. Select each run's checkpoint by validation fitness `0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95`; do not use the internal test set for selection.
5. Evaluate each selected checkpoint on the fixed internal test set with `scripts/val.py`, then aggregate the three values using sample mean and sample standard deviation.
6. For robustness, use `scripts/robustness_eval.py` on the same test images and unchanged labels. The reported conditions are 640→480→640, 640→320→640, brightness 0.6×, and brightness 1.4×.
7. For held-out uncropped furniture photographs, use `scripts/full_image_inference.py`. The combined strategy uses native-resolution 640×640 windows with stride 512 plus an aspect-ratio-preserving overview pass, followed by class-wise NMS at IoU 0.45.
8. For comparison models, consult `comparisons/model_registry.yaml` and use the exact upstream commit and architecture from the original run. Apply the shared split, seeds, input size, checkpoint-selection rule, and metric aggregation in `comparisons/shared_protocol.yaml`. Do not treat `comparisons/paper_reported_results.csv` as newly reproduced output.
9. Capture `scripts/system_info.py`, the source commit, data-manifest hash, configuration, command line, `args.yaml`, `results.csv`, best-epoch metadata, and checkpoint hash for every run. Fill `comparisons/measured_results_template.csv` only from these retained outputs.
10. Plot curves with `visualization/plot_training_curves.py`; never manufacture missing epochs. Run `scripts/audit_release.py` before archiving any public artifact.

## Model complexity verification

The proposed MFMD-YOLOv8 model contains 4.342 M parameters and requires 20.911 GFLOPs at an input resolution of 640×640 (exact parameter count: 4,342,313).
To verify model complexity without requiring weights or datasets:
```bash
python scripts/model_info.py
```
This executes `model.info(imgsz=640)` using standard Ultralytics / PyTorch profiling without dividing by 2 (reporting full 20.911 GFLOPs rather than the ~10.455 GMACs convention).

The manuscript reports the software environment as Windows 10 Pro, Python 3.10.19, PyTorch 2.5.1, CUDA 12.1, cuDNN 9.1, Ultralytics 8.0.166, AMP enabled, and an NVIDIA RTX 4060 Ti 8 GB. Runtime and FLOPs depend on the local hardware and software build.
