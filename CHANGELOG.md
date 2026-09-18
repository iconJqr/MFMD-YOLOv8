# Changelog

## 0.1.0-reconstruction

- Added the MFMD-YOLOv8, YOLOv8n, RCS-OSA, and BRA configurations.
- Added explicit CIoU/WIoU v3 and TaskAligned/legacy ATSS switches.
- Added three-seed, ablation, validation, robustness, and full-image scripts.
- Added the 12-model comparison registry and manuscript-transcribed results.
- Added reconstructed dataset preparation, offline augmentation, metadata, and
  interpretability utilities.
- Added evaluation, visualization, environment capture, model/data cards,
  synthetic smoke data, checksums, and automated tests.
- Documented model complexity: The proposed MFMD-YOLOv8 model contains 4.342 M parameters (4,342,313 parameters) and requires 20.911 GFLOPs at an input resolution of 640×640 (unified from rounded 20.9 to exact 20.911 GFLOPs, standard `model.info(imgsz=640)` convention).

This is a reconstruction-oriented release. It does not assert that missing
original runs, weights, or split identities have been recovered.
