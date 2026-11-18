import os

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.picai_losses import FocalAndCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

PICAI_VAL_DENSE_NNUNET_PREPROCESSED = "PICAI_VAL_DENSE_NNUNET_PREPROCESSED"


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


class nnUNetTrainerPICAI_FLCE_5epochs(nnUNetTrainerPICAI_FLCE):
    """Short-run trainer for end-to-end pipeline validation."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1


class _PicaiDenseValidationMixin:
    """Train on sparse preprocessed data; validate with dense labels when configured."""

    def _configure_picai_dense_validation_paths(self) -> None:
        dense_base = os.environ.get(PICAI_VAL_DENSE_NNUNET_PREPROCESSED)
        self.picai_val_dense_preprocessed_base = dense_base
        if dense_base:
            self.picai_val_dense_preprocessed_folder = join(
                dense_base, self.configuration_manager.data_identifier
            )
        else:
            self.picai_val_dense_preprocessed_folder = None

    @property
    def picai_uses_dense_validation(self) -> bool:
        return self.picai_val_dense_preprocessed_folder is not None

    def _validation_preprocessed_folder(self) -> str:
        if self.picai_uses_dense_validation:
            return self.picai_val_dense_preprocessed_folder
        return self.preprocessed_dataset_folder

    def _validation_preprocessed_folder_base(self) -> str:
        if self.picai_uses_dense_validation:
            return self.picai_val_dense_preprocessed_base
        return self.preprocessed_dataset_folder_base

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        dataset_tr = self.dataset_class(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )
        val_folder = self._validation_preprocessed_folder()
        dataset_val = self.dataset_class(
            val_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )
        return dataset_tr, dataset_val

    def perform_actual_validation(self, save_probabilities: bool = False):
        if not self.picai_uses_dense_validation:
            return super().perform_actual_validation(save_probabilities)

        sparse_folder = self.preprocessed_dataset_folder
        sparse_base = self.preprocessed_dataset_folder_base
        self.preprocessed_dataset_folder = self._validation_preprocessed_folder()
        self.preprocessed_dataset_folder_base = self._validation_preprocessed_folder_base()
        try:
            return super().perform_actual_validation(save_probabilities)
        finally:
            self.preprocessed_dataset_folder = sparse_folder
            self.preprocessed_dataset_folder_base = sparse_base


class nnUNetTrainerPICAI_FLCE_SparseSupervision(_PicaiDenseValidationMixin, nnUNetTrainerPICAI_FLCE):
    """Sparse training masks with fold validation on dense preprocessed labels."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._configure_picai_dense_validation_paths()
        if self.picai_uses_dense_validation:
            self.print_to_log_file(
                f"PI-CAI dense validation preprocessed data: "
                f"{self.picai_val_dense_preprocessed_folder}",
                also_print_to_console=True,
            )


class nnUNetTrainerPICAI_FLCE_SparseSupervision_5epochs(nnUNetTrainerPICAI_FLCE_SparseSupervision):
    """Short-run sparse-supervision trainer for smoke tests."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
