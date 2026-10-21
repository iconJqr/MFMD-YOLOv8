"""Validate the comparison-model registry and source metadata fields."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "faster_rcnn", "ssd_mobilenet_v2", "yolov5n", "yolov8n", "yolov8s", "yolov9t",
    "yolov10n", "yolov11n", "yolov12n", "rtmdet_tiny", "rt_detr_l", "rt_detr_x",
}
STATUSES = {"verified", "partially_verified", "candidate_entrypoint", "source_details_not_recorded"}
NOT_RECORDED = "not_recorded_in_release"


def main() -> None:
    registry = yaml.safe_load((ROOT / "model_registry.yaml").read_text(encoding="utf-8"))
    models = registry["models"]
    if set(models) != EXPECTED:
        raise ValueError(f"Registry mismatch: missing={EXPECTED - set(models)}, extra={set(models) - EXPECTED}")
    for key, model in models.items():
        missing = {field for field in ("paper_name", "status", "paper_params_m", "paper_gflops") if field not in model}
        if missing:
            raise ValueError(f"{key}: missing fields {sorted(missing)}")
        if model["status"] not in STATUSES:
            raise ValueError(f"{key}: invalid status {model['status']!r}")
        unresolved = any(NOT_RECORDED in str(model.get(field, "")).lower() for field in (
            "upstream_repository", "upstream_version_or_commit", "model_entrypoint", "optimizer"))
        if unresolved and model["status"] == "verified":
            raise ValueError(f"{key}: unresolved fields cannot be marked verified")
    print(f"validated {len(models)} comparison models")


if __name__ == "__main__":
    main()
