import csv
import math
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "Faster R-CNN", "SSD-MobileNetV2", "YOLOv5n", "YOLOv8n", "YOLOv8s", "YOLOv9t",
    "YOLOv10n", "YOLOv11n", "YOLOv12n", "RTMDet-tiny", "RT-DETR-L", "RT-DETR-X",
}


def test_registry_and_paper_results_cover_the_same_models():
    registry = yaml.safe_load((ROOT / "comparisons" / "model_registry.yaml").read_text(encoding="utf-8"))
    registry_names = {item["paper_name"] for item in registry["models"].values()}
    with (ROOT / "comparisons" / "paper_reported_results.csv").open(encoding="utf-8", newline="") as stream:
        result_names = {row["model"] for row in csv.DictReader(stream)}
    assert registry_names == EXPECTED
    assert result_names == EXPECTED


def test_measured_template_has_three_seeds_for_all_models_and_mfmd():
    with (ROOT / "comparisons" / "measured_results_template.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], set()).add(int(row["seed"]))
        assert row["provenance"] == "measured_run_pending"
    assert set(by_model) == EXPECTED | {"MFMD-YOLOv8"}
    assert all(seeds == {0, 1, 42} for seeds in by_model.values())


def test_three_seed_aggregation(tmp_path):
    source = tmp_path / "measured.csv"
    source.write_text(
        "model,seed,precision,recall,map50,map50_95,params_m,gflops,fps\n"
        "demo,0,0.7,0.6,0.8,0.5,3,8,100\n"
        "demo,1,0.8,0.7,0.9,0.6,3,8,101\n"
        "demo,42,0.9,0.8,1.0,0.7,3,8,102\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "comparisons" / "scripts" / "aggregate_three_seeds.py"),
         "--input", str(source), "--output", str(output)],
        check=True,
    )
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 1
    assert float(rows[0]["precision_mean"]) == 0.8
    assert math.isclose(float(rows[0]["precision_sd"]), 0.1)
