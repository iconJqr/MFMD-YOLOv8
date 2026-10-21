# Grad-CAM and Pointing Game

The scripts generate Grad-CAM maps and Pointing Game summaries from supplied model weights, images, and labels. Generated outputs are new measurements for the supplied inputs.

`generate_gradcam.py` creates class-specific P3/P4/P5 Grad-CAM arrays and an equal-weight fusion for a supplied YOLO-compatible checkpoint. `pointing_game.py` tests the peak of each image/class map against every matching-class ground-truth instance and reports class-wise and macro-average hit rates.

This implementation can directly load the released Ultralytics YOLO models. The manuscript's Pointing Game also reports Faster R-CNN and RTMDet-tiny; those external frameworks need their own checkpoint adapters and layer targets, which are not included because the original framework versions, weights, and CAM targets were not retained. Generated maps are new outputs, not restored manuscript figures.

Layer indices are model-configuration dependent. The defaults in `select_layers.yaml` correspond to the released baseline and MFMD YAMLs. Always verify them after changing an architecture. Generated heatmaps must be treated as newly generated results, not as recovery of the original figure.
