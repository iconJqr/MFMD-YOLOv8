# Dataset card: Mongolian furniture motif dataset

## Availability and scope

The source photographs and annotations are not distributed in this package.
Users must obtain them under the authors' data-access and cultural-property
conditions. No sample identifier in this repository is a recovered original
identifier.

## Manuscript-reported composition

| Split | Images | Animal instances | Plant instances | Geometry instances |
|---|---:|---:|---:|---:|
| Train, original | 1,051 | 583 | 1,026 | 784 |
| Validation | 138 | 68 | 132 | 101 |
| Internal test | 131 | 65 | 135 | 98 |

The three rows total 1,320 ROI images and 2,992 annotated instances. The
training-only offline process reportedly generated 3,153 additional images,
for 4,204 training images. These counts are transcribed from the manuscript,
not recalculated from data in this release.

## Format and partition protocol

YOLO labels use `class x_center y_center width height`, normalized to image
dimensions, with classes `0 geometry`, `1 animal`, `2 plant`. Partitioning must
be source-group-disjoint and must happen before augmentation. Validation and
test images must not be augmented. Use `dataset_tools/` to validate a local
copy and to produce a leakage report.

## Reconstructed materials

`dataset_metadata/split_manifest.template.csv` and
`configs/offline_augmentation_reconstructed.yaml` are reconstruction aids.
They do not recover the lost original partition. The augmentation exporter
records every generated item in `augmentation_audit.csv`; its exact operation
order and RandAugment mapping are reconstructed choices and are marked as such.

## Limitations

The manuscript does not preserve the image-level split identifiers,
photographic subject metadata, inter-annotator records, or a machine-readable
license. These remain `pending_author_confirmation` and must not be inferred.
