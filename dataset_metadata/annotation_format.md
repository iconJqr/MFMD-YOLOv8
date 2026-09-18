# Annotation and class format

The original annotations were created with LabelImg in Pascal VOC format and converted for each detector. The public training code uses YOLO text labels with one row per object:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`. The released class-index order is:

```text
0 geometry
1 animal
2 plant
```

The private image collection is not included. `split_manifest.example.csv` demonstrates the required anonymous metadata columns but contains illustrative identifiers rather than the lost original source mapping.
