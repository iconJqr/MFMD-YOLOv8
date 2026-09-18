# Training-curve visualization

The plotter accepts retained Ultralytics `results.csv` files and exports P, R,
mAP@0.5, and mAP@0.5:0.95 as PNG and SVG. Raw epoch fluctuations are shown as
transparent traces and the centered rolling mean as the main curve. Missing
logs are not reconstructed or interpolated.

```bash
python visualization/plot_training_curves.py \
  --run YOLOv8n=/path/to/yolov8n/results.csv \
  --run MFMD-YOLOv8=/path/to/mfmd/results.csv \
  --output figures/training --aspect 4:3 --smooth-window 11
```
