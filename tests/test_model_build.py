from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "config",
    ["yolov8n_baseline.yaml", "yolov8n_rcsosa.yaml", "yolov8n_bra.yaml", "mfmd_yolov8.yaml"],
)
def test_model_build_and_forward(config):
    model = YOLO(str(ROOT / "configs" / "models" / config), task="detect").model.eval()
    with torch.no_grad():
        prediction = model(torch.zeros(1, 3, 224, 224))
    assert prediction is not None


def test_paper_loss_and_assigner_configuration():
    model = YOLO(str(ROOT / "configs" / "models" / "mfmd_yolov8.yaml"), task="detect").model
    model.args = get_cfg(overrides={"iou_loss": "wiou_v3", "assigner": "task_aligned"})
    criterion = model.init_criterion()
    assert criterion.iou_loss == "wiou_v3"
    assert criterion.assigner_name == "task_aligned"
    assert criterion.use_dfl is True
