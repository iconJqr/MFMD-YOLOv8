# Author confirmation checklist

Do not convert an unchecked item into a public factual claim.

- [x] Model complexity confirmed: The proposed MFMD-YOLOv8 model contains 4.342 M parameters and requires 20.911 GFLOPs at an input resolution of 640×640 (exact parameter count: 4,342,313, profiled via standard `model.info(imgsz=640)`). Public documentation and records aligned.
- [ ] Resolve whether the final experiments used Ultralytics 8.1.34 or the
  bundled source version 8.0.166.
- [ ] Confirm whether final tables used TaskAlignedAssigner or legacy ATSS.
- [ ] Provide the original CIoU code/runs for every row labelled “without
  WIoU”, or rerun those controls with the explicit switch.
- [ ] Locate all three seed run directories (`args.yaml`, `results.csv`, and
  selected checkpoints) for the baseline, ablations, and final model.
- [ ] Confirm that EMA state and WIoU running statistics were reset between
  every sequential run.
- [ ] Supply the anonymous source-group-to-split manifest and leakage audit.
- [ ] Confirm the retained DFL setting and WIoU distance-term gradient behavior.
- [ ] Record immutable commits, exact model entries, pretrained sources, and
  optimizer settings for all 12 comparison models.
- [ ] Record the exact FPS timing scope, warm-up count, repetitions, batch size,
  precision mode, and device state.
- [ ] State the dataset access/licensing procedure and ethical/cultural-use
  restrictions without exposing private image identities.
- [ ] Replace the placeholder repository URL in `CITATION.cff`.
