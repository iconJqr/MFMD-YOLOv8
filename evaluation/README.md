# Evaluation utilities

- `select_best_checkpoint.py` applies the paper protocol to a real validation
  `results.csv`; the internal test set is never used for selection.
- `benchmark_model.py` writes measured metrics, timing scope, hardware, and
  software versions to JSON. Its inference-only FPS must not be mixed with
  end-to-end FPS.
- `full_image_evaluate.py` exports sliding-window/full-image predictions and an
  audit manifest. Quantitative scoring requires the retained full-image ground
  truth and a declared evaluator; those annotations are not present here.

These scripts contain no prefilled experimental measurements.
