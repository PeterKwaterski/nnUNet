import torch
from torch.nn import functional as F

from nnunetv2.training.loss.picai_losses import FocalAndCrossEntropyLoss, MulticlassFocalLoss


def test_focal_gamma_zero_matches_cross_entropy():
    logits = torch.tensor([[[[2.0, -1.0]], [[-2.0, 1.0]]]])
    target = torch.tensor([[[[0, 1]]]])

    actual = MulticlassFocalLoss(gamma=0)(logits, target)
    expected = F.cross_entropy(logits, target[:, 0])

    torch.testing.assert_close(actual, expected)


def test_focal_loss_rewards_better_predictions():
    target = torch.zeros((1, 1, 2, 2), dtype=torch.long)
    good = torch.stack((torch.full((1, 2, 2), 5.0), torch.full((1, 2, 2), -5.0)), dim=1)
    bad = -good

    loss = MulticlassFocalLoss()

    assert loss(good, target) < loss(bad, target)


def test_compound_loss_is_weighted_sum():
    logits = torch.tensor([[[[1.5, -0.5]], [[-1.5, 0.5]]]])
    target = torch.tensor([[[[0, 1]]]])
    focal_weight = 0.3
    loss = FocalAndCrossEntropyLoss(focal_weight=focal_weight)

    expected = focal_weight * loss.focal(logits, target)
    expected += (1 - focal_weight) * loss.cross_entropy(logits, target)

    torch.testing.assert_close(loss(logits, target), expected)


def test_ignore_index_excludes_voxels():
    logits = torch.tensor([[[[2.0, -100.0]], [[-2.0, 100.0]]]])
    target = torch.tensor([[[[0, -1]]]])

    actual = MulticlassFocalLoss(ignore_index=-1)(logits, target)
    expected = MulticlassFocalLoss()(logits[..., :1], target[..., :1])

    torch.testing.assert_close(actual, expected)
