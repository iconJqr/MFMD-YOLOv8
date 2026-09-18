# Paper-to-code map

| Manuscript element | Release location | Implementation detail |
|---|---|---|
| YOLOv8n baseline | `configs/models/yolov8n_baseline.yaml` | C2f backbone/neck, P3/P4/P5 detection |
| MFMD-YOLOv8 | `configs/models/mfmd_yolov8.yaml` | RCS-OSA + BRA (4.342 M params, 20.911 GFLOPs at 640×640) |
| RCS-OSA | `ultralytics/nn/modules/block.py` | `RCSOSA`, `SR`, RepConv-backed one-shot aggregation |
| BRA | `ultralytics/nn/modules/block.py` | `BiLevelRoutingAttention`, after SPPF |
| CIoU | `ultralytics/utils/loss.py` and `ultralytics/utils/metrics.py` | baseline and no-WIoU ablations |
| WIoU v3 | `ultralytics/utils/loss.py` and `ultralytics/utils/metrics.py` | non-monotonic focusing with γ=1.9, δ=3 |
| DFL | `ultralytics/utils/loss.py` | retained when `reg_max > 1`; default gain 1.5 |
| target assignment | `ultralytics/utils/loss.py` | paper default `task_aligned`, top-k 10, α=0.5, β=6.0 |
| training entrypoint | `scripts/train.py` | explicit loss and seed |
| three-seed protocol | `scripts/train_three_seeds.py` | seeds 0, 1, 42 |
| ablations | `configs/ablation.yaml`, `scripts/train_ablation.py` | eight rows from the manuscript |
| robustness evaluation | `scripts/robustness_eval.py` | original, 480, 320, 0.6×, 1.4× |
| full-image inference | `scripts/full_image_inference.py` | full, sliding, combined |
| comparison-model registry | `comparisons/model_registry.yaml` | 12 detectors with provenance status |
| comparison reference results | `comparisons/paper_reported_results.csv` | manuscript values, not regenerated values |
| three-seed aggregation | `comparisons/scripts/aggregate_three_seeds.py` | mean and sample SD |
| external comparison launcher | `comparisons/scripts/train_ultralytics_external.py` | explicit external repository and API |
| dataset preparation | `dataset_tools/` | reconstructed ROI, conversion, split, augmentation, validation, and leakage tools |
| training curves | `visualization/plot_training_curves.py` | plots only supplied real `results.csv` files |
| checkpoint selection | `evaluation/select_best_checkpoint.py` | validation fitness; never test-set selection |
| metrics and runtime | `evaluation/benchmark_model.py` | measured JSON with timing scope and environment |
| full-image prediction export | `evaluation/full_image_evaluate.py` | auditable tiled/full predictions; external scorer required |
| Grad-CAM/Pointing Game | `interpretability/` | reconstructed protocol requiring real weights/data |
| environment capture | `scripts/system_info.py` | captures actual run environment without secrets |
| release integrity | `scripts/audit_release.py`, `scripts/create_hash_manifest.py` | privacy/artifact audit and SHA-256 manifest |

## Implementation clarifications

- The proposed MFMD-YOLOv8 model contains 4.342 M parameters and requires 20.911 GFLOPs at an input resolution of 640×640 (exact parameter count: 4,342,313). Complexity profiling follows standard  without dividing by 2.

- CIoU is used whenever `iou_loss: ciou`; WIoU v3 is used only when `iou_loss: wiou_v3`. BCE and DFL remain active in both cases.
- WIoU's running mean is reset at the beginning of each loss object. The distance-term denominator follows the experimental implementation and is not detached; only the running-quality statistics used for focusing are detached.
- The public configuration defaults to TaskAlignedAssigner because the manuscript describes YOLOv8's task-aligned assignment. `assigner: atss` remains available only for reproducing legacy local experiments and is not the paper default.
- The exact training set and offline augmentation exporter are intentionally not included because the source images and annotations are private. The paper specifies the fixed offline set and its parameters.
