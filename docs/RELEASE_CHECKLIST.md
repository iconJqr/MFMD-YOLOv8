# Release checklist

- Included: source code for YOLOv8, RCS-OSA, BRA, CIoU/WIoU v3, DFL, model YAMLs, paper training settings, evaluation and inference scripts.
- Included: explicit configuration for baseline and the eight ablation combinations.
- Included: registry, shared protocol, reported table values, external launcher, and aggregation utility for all 12 comparison detectors.
- Included: dataset conversion/validation/leakage tools, reconstructed offline augmentation, metadata templates, and a dataset card.
- Included: training-curve plotting, checkpoint selection, measured benchmark export, full-image prediction export, Grad-CAM, and Pointing Game utilities.
- Included: model card, third-party notices, reported-environment record, CPU smoke Dockerfile, synthetic smoke dataset, CI tests, release audit, and checksum/archive tools.
- Excluded: complete third-party comparison repositories and their checkpoints; use the cited upstream licenses and immutable commits.
- Excluded: all image data, annotations, pretrained weights, `best.pt`/`last.pt`, `runs/`, caches, generated figures, and local absolute paths.
- Before uploading: complete `docs/AUTHOR_CONFIRMATION_CHECKLIST.md`, replace any repository URL placeholder in the manuscript and `CITATION.cff`, run tests and `scripts/audit_release.py`, generate `SHA256SUMS.txt`, and build the clean archive.
- Note on reproducibility: Model complexity is confirmed (The proposed MFMD-YOLOv8 model contains 4.342 M parameters and requires 20.911 GFLOPs at an input resolution of 640×640). Table reproduction requires matching the stated training protocol, assigner, and seed runs.
