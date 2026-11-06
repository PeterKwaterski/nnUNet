from typing import Optional, Sequence

import torch
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from torch import nn
from torch.nn import functional as F


class MulticlassFocalLoss(nn.Module):
    """Numerically stable multiclass focal loss for dense segmentation logits."""

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        self.ignore_index = ignore_index
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim:
            if target.shape[1] != 1:
                raise ValueError("target must have one label channel")
            target = target[:, 0]
        target = target.long()

        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0

        safe_target = torch.where(valid, target, 0)
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = torch.gather(log_probs, 1, safe_target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.class_weights is not None:
            if self.class_weights.numel() != logits.shape[1]:
                raise ValueError("class_weights must contain one value per output class")
            loss = loss * self.class_weights[safe_target]

        return loss[valid].mean()


class FocalAndCrossEntropyLoss(nn.Module):
    """PI-CAI's equally weighted focal-loss and cross-entropy objective."""

    def __init__(
        self,
        focal_weight: float = 0.5,
        gamma: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        if not 0 <= focal_weight <= 1:
            raise ValueError("focal_weight must be in [0, 1]")
        self.focal_weight = focal_weight
        weight = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.focal = MulticlassFocalLoss(gamma, class_weights, ignore_index)
        self.cross_entropy = RobustCrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal = self.focal(logits, target)
        cross_entropy = self.cross_entropy(logits, target)
        return self.focal_weight * focal + (1 - self.focal_weight) * cross_entropy
