# Synthetic example

The generator creates colored rectangles and YOLO labels solely to verify file
loading and model execution without distributing private images. It is not a
surrogate dataset, must never be mixed with the research dataset, and must not
be used for accuracy claims.

```bash
python examples/generate_synthetic_dataset.py --output synthetic_dataset
python examples/smoke_test.py
```
