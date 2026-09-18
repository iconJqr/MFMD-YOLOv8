# Comparison-model reproducibility

This directory documents the 12 comparison detectors reported in the manuscript. It does **not** vendor their complete third-party repositories or checkpoints. Each detector should be trained in an isolated environment created from its cited upstream repository, while using the same fixed furniture-source split and evaluation protocol.

## Included files

- `model_registry.yaml`: model identity, upstream project, candidate entrypoint, paper-reported complexity, and verification status.
- `shared_protocol.yaml`: dataset, training-seed, checkpoint-selection, and evaluation constraints common to all models.
- `paper_reported_results.csv`: the values transcribed from the manuscript; these are reference values, not newly generated outputs.
- `scripts/train_ultralytics_external.py`: launcher for comparison models implemented by an external Ultralytics checkout.
- `scripts/aggregate_three_seeds.py`: computes mean and sample standard deviation from three per-seed result rows.
- `scripts/validate_registry.py`: checks that all 12 comparison models are represented and that unresolved provenance remains explicitly marked.
- `THIRD_PARTY_CODE.md`: repository and licensing guidance.

## Important status distinction

`verified_from_manuscript` means that the manuscript directly states the value. `candidate_from_local_code` means that a matching model/configuration was found in a local code tree, but no final furniture-dataset run record established that it produced the manuscript table. `pending_author_confirmation` means that the exact implementation, version, or optimizer must be recovered from the original experiment before claiming exact reproducibility.

The release therefore separates two goals:

1. transparent disclosure of every detector reported in the paper; and
2. exact executable reproduction, which is possible only after the original repository commit, configuration, and three seed logs are confirmed.

## External Ultralytics models

Run the launcher with a clean upstream checkout. Do not point it at this MFMD fork unless reproducing YOLOv8n/MFMD ablations.

```bash
python comparisons/scripts/train_ultralytics_external.py \
  --repo /path/to/upstream/repository \
  --data /path/to/data.yaml \
  --model yolov8s.pt \
  --api YOLO \
  --optimizer SGD \
  --project runs/comparisons/yolov8s
```

The launcher runs seeds 0, 1, and 42 sequentially and records the imported package version. Replace the model identifier and optimizer only with values supported by the original experiment records.

Faster R-CNN, SSD-MobileNetV2, and RTMDet-tiny require their own upstream training APIs. Their exact implementation records are currently marked for author confirmation in `model_registry.yaml`; generic substitutes have deliberately not been presented as the models that generated the paper's results.
