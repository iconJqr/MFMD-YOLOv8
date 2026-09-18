# Grad-CAM and Pointing Game

The manuscript contains qualitative Grad-CAM figures and a Pointing Game summary. The original heatmap arrays are missing, so this directory provides a reproducible replacement implementation rather than fabricated heatmaps.

`generate_gradcam.py` creates class-specific P3/P4/P5 Grad-CAM arrays and an equal-weight fusion for a supplied checkpoint. `pointing_game.py` tests whether the strongest fused activation lies inside a ground-truth box of the same class.

Layer indices are model-configuration dependent. The defaults in `select_layers.yaml` correspond to the released baseline and MFMD YAMLs. Always verify them after changing an architecture. Generated heatmaps must be treated as newly generated results, not as recovery of the original figure.
