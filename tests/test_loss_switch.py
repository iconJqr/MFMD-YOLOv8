import torch

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import WIoU_Scale


def inputs():
    predicted = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]], requires_grad=True)
    target = torch.tensor([[[0.2, 0.1, 2.1, 2.0]]])
    score = torch.ones(1, 1, 1)
    mask = torch.tensor([[True]])
    return predicted, target, score, mask


def test_ciou_and_wiou_v3_are_finite_and_differentiable():
    values = []
    for name in ("ciou", "wiou_v3"):
        predicted, target, score, mask = inputs()
        loss, _ = BboxLoss(reg_max=15, use_dfl=False, iou_loss=name)(
            torch.empty(1, 1, 64), predicted, torch.zeros(1, 2), target, score, score.sum(), mask)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(predicted.grad).all()
        values.append(loss.detach())
    assert not torch.allclose(values[0], values[1])


def test_wiou_running_mean_can_be_reset():
    WIoU_Scale.iou_mean = 0.25
    WIoU_Scale.reset()
    assert WIoU_Scale.iou_mean == 1.0
