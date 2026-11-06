import numpy as np

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.picai_losses import FocalAndCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerPICAI_FLCE(nnUNetTrainer):
    """nnU-Net v2 trainer using the PI-CAI baseline's 50/50 focal + CE loss."""

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("PI-CAI FL+CE trainer does not support region-based training")

        loss = FocalAndCrossEntropyLoss(
            focal_weight=0.5,
            gamma=2.0,
            ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100,
        )

        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
