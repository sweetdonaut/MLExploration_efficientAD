"""
Image filter using ONNX Runtime for inference.
Filters out abnormal images from a set of 3 similar images in a .tif file.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
import tifffile


class ImageFilterOnnx:
    def __init__(
        self,
        onnx_path: str,
        similarity_threshold: float = 0.7,
        target_size: tuple = (256, 256),
        use_gpu: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.target_size = target_size

        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.active_provider = self.session.get_providers()[0]

    def load_tif(self, tif_path: str) -> np.ndarray:
        """Load .tif file containing 3 images."""
        images = tifffile.imread(tif_path)
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        return images

    def preprocess(self, images: np.ndarray) -> np.ndarray:
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
        features_norm = features / (norm + 1e-8)
        similarity = features_norm @ features_norm.T
        return similarity

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

    parser = argparse.ArgumentParser(description='Filter abnormal images using ONNX')
    parser.add_argument('tif_path', type=str, help='Path to .tif file')
    parser.add_argument('--onnx', type=str, default='API_filter/feature_extractor.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Similarity threshold (default: 0.7)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    onnx_path = project_root / args.onnx

    filter_instance = ImageFilterOnnx(
        onnx_path=str(onnx_path),
        similarity_threshold=args.threshold,
        use_gpu=not args.cpu,
    )

    print(f"Provider: {filter_instance.active_provider}")

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
