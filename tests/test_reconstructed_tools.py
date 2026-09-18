import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset_tools"))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "examples"))

from generate_synthetic_dataset import generate
from offline_augmentation import filter_boxes, load_boxes, mosaic
from select_best_checkpoint import select


def test_synthetic_dataset_and_labels(tmp_path):
    data = generate(tmp_path / "toy", size=64)
    assert data.is_file()
    images = list((tmp_path / "toy" / "images" / "train").glob("*.png"))
    assert len(images) == 8
    image = cv2.imread(str(images[0]))
    boxes = load_boxes(tmp_path / "toy" / "labels" / "train" / f"{images[0].stem}.txt", 64, 64)
    assert image.shape == (64, 64, 3)
    assert boxes.shape == (1, 5)


def test_box_filter_uses_retained_area():
    before = np.array([[0, 0, 0, 100, 100]], dtype=np.float32)
    assert len(filter_boxes(before, np.array([[0, 0, 0, 50, 50]], dtype=np.float32), 100, 100)) == 1
    assert len(filter_boxes(before, np.array([[0, 0, 0, 10, 10]], dtype=np.float32), 100, 100)) == 0


def test_mosaic_boxes_stay_on_canvas(tmp_path):
    generate(tmp_path / "toy", size=640)
    images = sorted((tmp_path / "toy" / "images" / "train").glob("*.png"))
    import random
    _, boxes, sources = mosaic(images[0], images[1:], tmp_path / "toy", random.Random(42))
    assert len(sources) == 4
    assert np.all(boxes[:, 1:] >= 0) and np.all(boxes[:, 1:] <= 640)


def test_validation_checkpoint_selection():
    frame = pd.DataFrame({"epoch": [0, 1, 2], "metrics/mAP50(B)": [0.6, 0.8, 0.9],
                          "metrics/mAP50-95(B)": [0.4, 0.7, 0.65]})
    epoch, score = select(frame, "fitness")
    assert epoch == 1
    assert score == 0.71


def test_leakage_checker(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_group_id", "split"))
        writer.writeheader()
        writer.writerows(({"source_group_id": "A", "split": "train"},
                          {"source_group_id": "B", "split": "test"}))
    completed = subprocess.run([sys.executable, str(ROOT / "dataset_tools" / "check_dataset_leakage.py"),
                                "--manifest", str(manifest)], capture_output=True, text=True)
    assert completed.returncode == 0


def test_training_curve_plotter_uses_real_csv(tmp_path):
    results = tmp_path / "results.csv"
    pd.DataFrame({
        "epoch": [0, 1, 2],
        "metrics/precision(B)": [0.2, 0.4, 0.5],
        "metrics/recall(B)": [0.3, 0.45, 0.55],
        "metrics/mAP50(B)": [0.2, 0.5, 0.6],
        "metrics/mAP50-95(B)": [0.1, 0.3, 0.4],
    }).to_csv(results, index=False)
    output = tmp_path / "curves"
    subprocess.run([sys.executable, str(ROOT / "visualization" / "plot_training_curves.py"),
                    "--run", f"demo={results}", "--output", str(output), "--smooth-window", "1"], check=True)
    assert (output / "precision.svg").is_file()
    assert (output / "map50_95.png").is_file()
