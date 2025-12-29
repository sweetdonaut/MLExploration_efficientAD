"""
Convert MediumPatchDescriptionNetwork to ONNX and run inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path


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


class FeatureExtractor(nn.Module):
    """Wrapper that includes GAP for ONNX export."""
    def __init__(self, pdn: MediumPatchDescriptionNetwork):
        super().__init__()
        self.pdn = pdn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.pdn(x)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.squeeze(-1).squeeze(-1)
        return features


def convert_to_onnx(weights_path: str, output_path: str, input_size: int = 256):
    """Convert PyTorch model to ONNX."""
    print(f"Loading weights from {weights_path}...")

    pdn = MediumPatchDescriptionNetwork(out_channels=384, padding=False)
    state_dict = torch.load(weights_path, map_location='cpu')
    pdn.load_state_dict(state_dict)
    pdn.eval()

    model = FeatureExtractor(pdn)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"Exporting to ONNX: {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")

    return output_path


class OnnxImageFilter:
    """Image filter using ONNX Runtime."""

    def __init__(
        self,
        onnx_path: str,
        similarity_threshold: float = 0.7,
        target_size: tuple = (256, 256),
    ):
        self.similarity_threshold = similarity_threshold
        self.target_size = target_size

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        print(f"ONNX Runtime provider: {self.session.get_providers()}")

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """Preprocess images for ONNX inference."""
        from PIL import Image

        processed = []
        for img in images:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            pil_img = Image.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize(self.target_size, Image.BILINEAR)
            img_array = np.array(pil_img).astype(np.float32) / 255.0

            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)

            img_tensor = np.transpose(img_array, (2, 0, 1))
            processed.append(img_tensor)

        return np.stack(processed).astype(np.float32)

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract features using ONNX Runtime."""
        outputs = self.session.run(None, {self.input_name: images})
        return outputs[0]

    def compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity."""
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        features_norm = features / norm
        similarity = features_norm @ features_norm.T
        return similarity

    def filter_images(self, tif_path: str) -> dict:
        """Filter images from a .tif file."""
        import tifffile

        images = tifffile.imread(tif_path)
        if images.ndim == 3:
            images = images[np.newaxis, ...]

        n_images = len(images)
        if n_images != 3:
            return {
                'keep': list(range(n_images)) if n_images < 3 else [],
                'discard': [] if n_images < 3 else list(range(n_images)),
                'similarities': None,
                'status': 'invalid_count',
            }

        tensors = self.preprocess(images)
        features = self.extract_features(tensors)
        sim_matrix = self.compute_similarity_matrix(features)

        sim_01, sim_02, sim_12 = sim_matrix[0, 1], sim_matrix[0, 2], sim_matrix[1, 2]
        similarities = {'0-1': float(sim_01), '0-2': float(sim_02), '1-2': float(sim_12)}

        threshold = self.similarity_threshold
        high_sim = [sim_01 >= threshold, sim_02 >= threshold, sim_12 >= threshold]

        if all(high_sim):
            return {'keep': [0, 1, 2], 'discard': [], 'similarities': similarities, 'status': 'ok'}
        if sum(high_sim) == 0:
            return {'keep': [], 'discard': [0, 1, 2], 'similarities': similarities, 'status': 'all_bad'}

        if high_sim[0] and not high_sim[1] and not high_sim[2]:
            return {'keep': [0, 1], 'discard': [2], 'similarities': similarities, 'status': 'one_bad'}
        elif high_sim[1] and not high_sim[0] and not high_sim[2]:
            return {'keep': [0, 2], 'discard': [1], 'similarities': similarities, 'status': 'one_bad'}
        elif high_sim[2] and not high_sim[0] and not high_sim[1]:
            return {'keep': [1, 2], 'discard': [0], 'similarities': similarities, 'status': 'one_bad'}

        avg_sim = [(sim_01 + sim_02) / 2, (sim_01 + sim_12) / 2, (sim_02 + sim_12) / 2]
        worst_idx = int(np.argmin(avg_sim))
        keep_indices = [i for i in range(3) if i != worst_idx]
        return {'keep': keep_indices, 'discard': [worst_idx], 'similarities': similarities, 'status': 'one_bad'}


def main():
    project_root = Path(__file__).parent.parent
    weights_path = project_root / "pre_trained/efficientad_pretrained_weights/pretrained_teacher_medium.pth"
    onnx_path = Path(__file__).parent / "feature_extractor.onnx"

    print("=" * 60)
    print("Step 1: Convert to ONNX")
    print("=" * 60)
    convert_to_onnx(str(weights_path), str(onnx_path))

    print("\n" + "=" * 60)
    print("Step 2: Test ONNX Inference")
    print("=" * 60)

    onnx_filter = OnnxImageFilter(str(onnx_path), similarity_threshold=0.7)

    test_dir = Path(__file__).parent / "broken_datasets"
    test_files = sorted(test_dir.glob("*.tif"))

    print(f"\n{'File':<35} {'Status':<10} {'Keep':<12} {'Discard'}")
    print("-" * 75)

    for f in test_files:
        result = onnx_filter.filter_images(str(f))
        print(f"{f.name:<35} {result['status']:<10} {str(result['keep']):<12} {result['discard']}")


if __name__ == "__main__":
    main()
