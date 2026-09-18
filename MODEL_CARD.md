# Model card: MFMD-YOLOv8

## Intended use

MFMD-YOLOv8 detects three Mongolian furniture motif categories: `geometry`,
`animal`, and `plant`. It is a research model for the dataset and evaluation
protocol described in the accompanying manuscript, not a general cultural
heritage recognition system.

## Architecture

The released configuration starts from YOLOv8n, replaces selected C2f blocks
with RCS-OSA, places Bi-Level Routing Attention after SPPF, and uses a
three-scale P3/P4/P5 detection head. BCE classification and DFL are retained;
box regression is configurable as CIoU or WIoU v3. See
`configs/models/mfmd_yolov8.yaml` and `docs/METHOD_CODE_MAP.md`.

## Model complexity and reported evaluation

The proposed MFMD-YOLOv8 model contains 4.342 M parameters and requires 20.911 GFLOPs at an input resolution of 640×640 (exact parameter count: 4,342,313, profiled via standard `model.info(imgsz=640)` without halving for GMACs).

Paper-transcribed comparison values are kept in
`comparisons/paper_reported_results.csv` with explicit provenance. They are
transcribed from the manuscript tables. Model weights and original run logs
should be obtained or evaluated locally.

## Limitations and risks

- Performance outside the photographed furniture, acquisition conditions, and
  three classes has not been established.
- Similar motifs, occlusion, small instances, illumination shifts, and domain
  changes may produce false positives or missed detections.
- Cultural labels are task-specific visual categories and must not be treated
  as provenance, authenticity, ethnicity, or historical attribution.
- The final training code version, target assigner used for reported tables,
  and CIoU controls remain pending author confirmation.

## License and citation

The bundled Ultralytics-derived code retains AGPL-3.0. See `LICENSE`,
`CITATION.cff`, and `THIRD_PARTY_NOTICES.md`. Dataset rights are separate and
the private images are not licensed by this source release.
