"""
Image filter prototype using pretrained MediumPatchDescriptionNetwork.
Filters out abnormal images from a set of 3 similar images in a .tif file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import tifffile


def imagenet_norm_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    return (x - mean) / std


class MediumPatchDescriptionNetwork(nn.Module):
    def __init__(self, out_channels: int = 384, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)


class ImageFilter:
    def __init__(
        self,
        weights_path: str,
        similarity_threshold: float = 0.7,
        device: str = None,
        target_size: tuple = (256, 256),
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold
        self.target_size = target_size

        self.model = MediumPatchDescriptionNetwork(out_channels=384, padding=False)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def load_tif(self, tif_path: str) -> np.ndarray:
        """Load .tif file containing 3 images."""
        images = tifffile.imread(tif_path)
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        return images

    def preprocess(self, images: np.ndarray) -> torch.Tensor:
        """Preprocess images: resize and normalize to [0, 1]."""
        processed = []
        for img in images:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            pil_img = Image.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize(self.target_size, Image.BILINEAR)
            img_array = np.array(pil_img).astype(np.float32) / 255.0

            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)

            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            processed.append(img_tensor)

        return torch.stack(processed).to(self.device)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features using the pretrained network."""
        with torch.no_grad():
            features = self.model(images)
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        return features

    def compute_similarity_matrix(self, features: torch.Tensor) -> np.ndarray:
        """Compute pairwise cosine similarity."""
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())
        return similarity.cpu().numpy()

    def filter_images(self, tif_path: str) -> dict:
        """
        Filter images from a .tif file.

        Returns:
            dict with keys:
                - 'keep': list of indices to keep
                - 'discard': list of indices to discard
                - 'similarities': pairwise similarity scores
                - 'status': 'ok', 'one_bad', or 'all_bad'
        """
        images = self.load_tif(tif_path)
        n_images = len(images)

        if n_images != 3:
            return {
                'keep': list(range(n_images)) if n_images < 3 else [],
                'discard': [] if n_images < 3 else list(range(n_images)),
                'similarities': None,
                'status': 'invalid_count',
                'message': f'Expected 3 images, got {n_images}'
            }

        tensors = self.preprocess(images)
        features = self.extract_features(tensors)
        sim_matrix = self.compute_similarity_matrix(features)

        sim_01 = sim_matrix[0, 1]
        sim_02 = sim_matrix[0, 2]
        sim_12 = sim_matrix[1, 2]

        similarities = {
            '0-1': float(sim_01),
            '0-2': float(sim_02),
            '1-2': float(sim_12),
        }

        threshold = self.similarity_threshold
        high_sim = [
            sim_01 >= threshold,
            sim_02 >= threshold,
            sim_12 >= threshold,
        ]

        if all(high_sim):
            return {
                'keep': [0, 1, 2],
                'discard': [],
                'similarities': similarities,
                'status': 'ok',
            }

        if sum(high_sim) == 0:
            return {
                'keep': [],
                'discard': [0, 1, 2],
                'similarities': similarities,
                'status': 'all_bad',
            }

        if high_sim[0] and not high_sim[1] and not high_sim[2]:
            return {
                'keep': [0, 1],
                'discard': [2],
                'similarities': similarities,
                'status': 'one_bad',
            }
        elif high_sim[1] and not high_sim[0] and not high_sim[2]:
            return {
                'keep': [0, 2],
                'discard': [1],
                'similarities': similarities,
                'status': 'one_bad',
            }
        elif high_sim[2] and not high_sim[0] and not high_sim[1]:
            return {
                'keep': [1, 2],
                'discard': [0],
                'similarities': similarities,
                'status': 'one_bad',
            }

        avg_sim = [
            (sim_01 + sim_02) / 2,
            (sim_01 + sim_12) / 2,
            (sim_02 + sim_12) / 2,
        ]
        worst_idx = int(np.argmin(avg_sim))
        keep_indices = [i for i in range(3) if i != worst_idx]

        return {
            'keep': keep_indices,
            'discard': [worst_idx],
            'similarities': similarities,
            'status': 'one_bad',
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Filter abnormal images from .tif file')
    parser.add_argument('tif_path', type=str, help='Path to .tif file')
    parser.add_argument('--weights', type=str,
                        default='pre_trained/efficientad_pretrained_weights/pretrained_teacher_medium.pth',
                        help='Path to pretrained weights')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Similarity threshold (default: 0.7)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    weights_path = project_root / args.weights

    filter_instance = ImageFilter(
        weights_path=str(weights_path),
        similarity_threshold=args.threshold,
    )

    result = filter_instance.filter_images(args.tif_path)

    print(f"\n{'='*50}")
    print(f"File: {args.tif_path}")
    print(f"{'='*50}")
    print(f"Status: {result['status']}")
    print(f"Keep indices: {result['keep']}")
    print(f"Discard indices: {result['discard']}")
    if result['similarities']:
        print(f"\nSimilarities:")
        for pair, sim in result['similarities'].items():
            print(f"  Image {pair}: {sim:.4f}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
