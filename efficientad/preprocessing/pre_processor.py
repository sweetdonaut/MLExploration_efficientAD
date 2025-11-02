# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pre-processing module for anomaly detection pipelines.

This module provides functionality for pre-processing data before model training
and inference through the :class:`PreProcessor` class.
"""

import torch
from lightning import Callback, LightningModule, Trainer
from torch import nn
from torchvision.transforms.v2 import Transform

from efficientad.data import Batch

from .utils.transform import get_exportable_transform


class PreProcessor(nn.Module, Callback):
    """Anomalib pre-processor.

    This class serves as both a PyTorch module and a Lightning callback, handling
    the application of transforms to data batches as a pre-processing step.

    Args:
        transform (Transform | None): Transform to apply to the data before passing it to the model.
    """

    def __init__(
        self,
        transform: Transform | None = None,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.export_transform = get_exportable_transform(self.transform)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Apply transforms to the batch of tensors during training."""
        del trainer, pl_module, batch_idx  # Unused
        if self.transform:
            batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Apply transforms to the batch of tensors during validation."""
        del trainer, pl_module, batch_idx  # Unused
        if self.transform:
            batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the batch of tensors during testing."""
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused
        if self.transform:
            batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the batch of tensors during prediction."""
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused
        if self.transform:
            batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply transforms to the batch of tensors for inference.

        This forward-pass is only used after the model is exported.
        Within the Lightning training/validation/testing loops, the transforms are
        applied in the ``on_*_batch_start`` methods.

        Args:
            batch (torch.Tensor): Input batch to transform.

        Returns:
            torch.Tensor: Transformed batch.
        """
        return self.export_transform(batch) if self.export_transform else batch
