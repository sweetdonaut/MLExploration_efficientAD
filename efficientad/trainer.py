# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure PyTorch trainer for EfficientAD.

This module provides a standalone trainer for EfficientAD that does not depend on
PyTorch Lightning or Anomalib infrastructure. It extracts and implements the core
training logic in pure PyTorch.
"""

import logging
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import CenterCrop, Compose, RandomGrayscale, Resize, ToImage, ToDtype

from efficientad.data import DownloadInfo, download_and_extract
from efficientad.models.torch_model import EfficientAdModel, EfficientAdModelSize, reduce_tensor_elems

logger = logging.getLogger(__name__)

# Download info for ImageNette dataset
IMAGENETTE_DOWNLOAD_INFO = DownloadInfo(
    name="imagenette2.tgz",
    url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    hashsum="6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd",
)

# Download info for pretrained teacher weights
WEIGHTS_DOWNLOAD_INFO = DownloadInfo(
    name="efficientad_pretrained_weights.zip",
    url="https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip",
    hashsum="c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a",
)


class EfficientADTrainer:
    """Pure PyTorch trainer for EfficientAD.

    This trainer implements the complete EfficientAD training pipeline without
    requiring PyTorch Lightning or Anomalib infrastructure.

    Args:
        model_size (EfficientAdModelSize | str): Size of the model ('small' or 'medium')
        teacher_out_channels (int): Number of teacher output channels. Defaults to 384.
        imagenet_dir (Path | str): Directory for ImageNette dataset. Defaults to "./datasets/imagenette".
        lr (float): Learning rate. Defaults to 0.0001.
        weight_decay (float): Weight decay for optimizer. Defaults to 0.00001.
        padding (bool): Whether to use padding in conv layers. Defaults to False.
        pad_maps (bool): Whether to pad output maps. Defaults to True.
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.
    """

    def __init__(
        self,
        model_size: EfficientAdModelSize | str = EfficientAdModelSize.S,
        teacher_out_channels: int = 384,
        imagenet_dir: Path | str = "./datasets/imagenette",
        lr: float = 0.0001,
        weight_decay: float = 0.00001,
        padding: bool = False,
        pad_maps: bool = True,
        device: str | None = None,
    ):
        self.imagenet_dir = Path(imagenet_dir)
        if not isinstance(model_size, EfficientAdModelSize):
            model_size = EfficientAdModelSize(model_size)
        self.model_size = model_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = EfficientAdModel(
            teacher_out_channels=teacher_out_channels,
            model_size=model_size,
            padding=padding,
            pad_maps=pad_maps,
        ).to(self.device)

        # Will be initialized during training
        self.optimizer = None
        self.scheduler = None
        self.imagenet_loader = None
        self.imagenet_iterator = None

    def prepare_pretrained_model(self) -> None:
        """Download and load pretrained teacher model weights."""
        from efficientad.models.torch_model import EfficientAdModelSize

        pretrained_models_dir = Path("./pre_trained/")
        if not (pretrained_models_dir / "efficientad_pretrained_weights").is_dir():
            download_and_extract(pretrained_models_dir, WEIGHTS_DOWNLOAD_INFO)

        model_size_str = self.model_size.value

        if self.model_size == EfficientAdModelSize.DINO:
            teacher_path = pretrained_models_dir / "efficientad_pretrained_weights" / "vit_small_patch16_dinov3.lvd1689m.safetensors"
            logger.info(f"Loading DINOv3 teacher model from {teacher_path}")

            from safetensors import safe_open
            state_dict = {}
            with safe_open(teacher_path, framework="pt", device=str(self.device)) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

            self.model.teacher.load_state_dict(state_dict, strict=True)
        else:
            teacher_path = pretrained_models_dir / "efficientad_pretrained_weights" / f"pretrained_teacher_{model_size_str}.pth"
            logger.info(f"Loading pretrained teacher model from {teacher_path}")
            self.model.teacher.load_state_dict(
                torch.load(teacher_path, map_location=torch.device(self.device), weights_only=True),
            )

        # Freeze teacher parameters
        for param in self.model.teacher.parameters():
            param.requires_grad = False
        self.model.teacher.eval()
        logger.info("✓ Teacher parameters frozen (requires_grad=False)")

    def prepare_imagenette_data(self, image_size: tuple[int, int]) -> None:
        """Prepare ImageNette dataset for training.

        Args:
            image_size (tuple[int, int]): Target image size (H, W).
        """
        data_transforms_imagenet = Compose([
            Resize((image_size[0] * 2, image_size[1] * 2)),
            RandomGrayscale(p=0.3),
            CenterCrop((image_size[0], image_size[1])),
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ])

        if not self.imagenet_dir.is_dir():
            download_and_extract(self.imagenet_dir, IMAGENETTE_DOWNLOAD_INFO)

        imagenet_dataset = ImageFolder(self.imagenet_dir, transform=data_transforms_imagenet)
        self.imagenet_loader = DataLoader(imagenet_dataset, batch_size=1, shuffle=True, pin_memory=True)
        self.imagenet_iterator = iter(self.imagenet_loader)

    @torch.no_grad()
    def compute_teacher_mean_std(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Calculate channel-wise mean and std of teacher model outputs.

        Args:
            dataloader (DataLoader): Training dataloader.

        Returns:
            dict[str, torch.Tensor]: Dictionary with 'mean' and 'std' tensors.
        """
        self.model.eval()
        n = None
        channel_sum = None
        channel_sum_sqr = None

        for batch in tqdm.tqdm(dataloader, desc="Computing teacher statistics"):
            images = batch['image'].to(self.device) if isinstance(batch, dict) else batch[0].to(self.device)
            y = self.model.teacher(images)

            if n is None:
                _, num_channels, _, _ = y.shape
                n = torch.zeros((num_channels,), dtype=torch.int64, device=y.device)
                channel_sum = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                channel_sum_sqr = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)

            n += y[:, 0].numel()
            channel_sum += torch.sum(y, dim=[0, 2, 3])
            channel_sum_sqr += torch.sum(y**2, dim=[0, 2, 3])

        channel_mean = channel_sum / n
        channel_std = torch.sqrt((channel_sum_sqr / n) - (channel_mean**2))

        return {
            "mean": channel_mean.float()[None, :, None, None],
            "std": channel_std.float()[None, :, None, None],
        }

    @torch.no_grad()
    def compute_validation_quantiles(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Calculate quantiles from validation set for normalization.

        Args:
            dataloader (DataLoader): Validation dataloader.

        Returns:
            dict[str, torch.Tensor]: Dictionary with quantile values.
        """
        self.model.eval()
        maps_st = []
        maps_ae = []

        logger.info("Computing validation quantiles")
        for batch in tqdm.tqdm(dataloader, desc="Computing validation quantiles"):
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                labels = batch.get('label', batch.get('gt_label', torch.zeros(len(images))))
            else:
                images, labels = batch[0].to(self.device), batch[1]

            # Only use normal images (label == 0)
            for img, label in zip(images, labels):
                if label == 0:
                    img = img.unsqueeze(0)  # Add batch dimension
                    map_st, map_ae = self.model.get_maps(img, normalize=False)
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)

        # Compute quantiles
        maps_st_flat = reduce_tensor_elems(torch.cat(maps_st))
        maps_ae_flat = reduce_tensor_elems(torch.cat(maps_ae))

        qa_st = torch.quantile(maps_st_flat, q=0.9).to(self.device)
        qb_st = torch.quantile(maps_st_flat, q=0.995).to(self.device)
        qa_ae = torch.quantile(maps_ae_flat, q=0.9).to(self.device)
        qb_ae = torch.quantile(maps_ae_flat, q=0.995).to(self.device)

        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def setup_optimizer(self, max_steps: int) -> None:
        """Setup optimizer and learning rate scheduler.

        Args:
            max_steps (int): Total number of training steps.
        """
        self.optimizer = torch.optim.Adam(
            list(self.model.student.parameters()) + list(self.model.ae.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning rate decay at 95% of training
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(0.95 * max_steps),
            gamma=0.1
        )

    def train_step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform single training step.

        Args:
            batch: Training batch (dict or tuple).

        Returns:
            tuple: (loss_st, loss_ae, loss_stae) - three loss components.
        """
        self.model.train()

        # Get batch images
        if isinstance(batch, dict):
            images = batch['image'].to(self.device)
        else:
            images = batch[0].to(self.device)

        # Get ImageNet batch
        try:
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
        except StopIteration:
            self.imagenet_iterator = iter(self.imagenet_loader)
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)

        # Forward pass
        loss_st, loss_ae, loss_stae = self.model(batch=images, batch_imagenet=batch_imagenet)

        # Backward pass
        total_loss = loss_st + loss_ae + loss_stae
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss_st, loss_ae, loss_stae

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation.

        Args:
            dataloader (DataLoader): Validation dataloader.

        Returns:
            dict: Validation metrics.
        """
        self.model.eval()

        anomaly_scores = []
        labels = []

        for batch in tqdm.tqdm(dataloader, desc="Validating"):
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                batch_labels = batch.get('label', batch.get('gt_label'))
            else:
                images, batch_labels = batch[0].to(self.device), batch[1]

            # Get predictions
            predictions = self.model(images)
            anomaly_scores.append(predictions.pred_score.cpu())
            labels.append(batch_labels)

        # Concatenate all results
        anomaly_scores = torch.cat(anomaly_scores)
        labels = torch.cat(labels)

        # Simple metrics
        mean_score = anomaly_scores.mean().item()

        return {
            "mean_anomaly_score": mean_score,
            "num_samples": len(labels),
        }

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        max_epochs: int = 70,
    ) -> None:
        """Train the model.

        Args:
            train_dataloader (DataLoader): Training data.
            val_dataloader (DataLoader): Validation data.
            max_epochs (int): Number of epochs to train. Defaults to 70.

        Raises:
            ValueError: If batch_size is not 1.
        """
        # Validate batch size
        if train_dataloader.batch_size != 1:
            msg = "train_batch_size for EfficientAd should be 1."
            raise ValueError(msg)

        logger.info("✓ Batch size validation passed (batch_size=1)")

        # Setup
        logger.info("Preparing pretrained teacher model...")
        self.prepare_pretrained_model()

        # Get image size from first batch
        first_batch = next(iter(train_dataloader))
        if isinstance(first_batch, dict):
            image_size = first_batch['image'].shape[-2:]
        else:
            image_size = first_batch[0].shape[-2:]

        logger.info(f"Image size: {image_size}")
        logger.info("Preparing ImageNette dataset...")
        self.prepare_imagenette_data(image_size)

        # Compute teacher statistics
        if not self.model.is_set(self.model.mean_std):
            logger.info("Computing teacher channel statistics...")
            mean_std = self.compute_teacher_mean_std(train_dataloader)
            self.model.mean_std.update(mean_std)

        # Setup optimizer
        max_steps = max_epochs * len(train_dataloader)
        self.setup_optimizer(max_steps)

        # Training loop
        logger.info(f"Starting training for {max_epochs} epochs...")
        for epoch in range(max_epochs):
            # Training
            epoch_losses = {"st": [], "ae": [], "stae": []}

            pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
            for batch in pbar:
                loss_st, loss_ae, loss_stae = self.train_step(batch)

                epoch_losses["st"].append(loss_st.item())
                epoch_losses["ae"].append(loss_ae.item())
                epoch_losses["stae"].append(loss_stae.item())

                # Update progress bar
                pbar.set_postfix({
                    "loss_st": f"{loss_st.item():.4f}",
                    "loss_ae": f"{loss_ae.item():.4f}",
                    "loss_stae": f"{loss_stae.item():.4f}",
                })

            # Log epoch statistics
            avg_loss_st = sum(epoch_losses["st"]) / len(epoch_losses["st"])
            avg_loss_ae = sum(epoch_losses["ae"]) / len(epoch_losses["ae"])
            avg_loss_stae = sum(epoch_losses["stae"]) / len(epoch_losses["stae"])

            logger.info(
                f"Epoch {epoch+1} - "
                f"loss_st: {avg_loss_st:.4f}, "
                f"loss_ae: {avg_loss_ae:.4f}, "
                f"loss_stae: {avg_loss_stae:.4f}"
            )

        logger.info("Training completed!")

        # Compute quantiles AFTER training completes (using final trained model)
        # This matches Anomalib's on_validation_start behavior
        logger.info("Computing validation quantiles with final trained model...")
        quantiles = self.compute_validation_quantiles(val_dataloader)
        self.model.quantiles.update(quantiles)
        logger.info("✓ Quantiles computed and stored")

    def save_model(self, path: Path | str) -> None:
        """Save model checkpoint.

        Args:
            path (Path | str): Path to save checkpoint.
        """
        state_dict = self.model.state_dict()
        state_dict_with_prefix = {f'model.{k}': v for k, v in state_dict.items()}

        torch.save({'state_dict': state_dict_with_prefix}, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path | str) -> None:
        """Load model checkpoint.

        Args:
            path (Path | str): Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        logger.info(f"Model loaded from {path}")
