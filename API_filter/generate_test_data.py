"""
Generate test dataset with broken images for testing the image filter.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import tifffile
import random

CROP_SIZE = 540
CENTER_REGION = 800
MAX_OFFSET = (CENTER_REGION - CROP_SIZE) // 2


def get_random_offsets(n: int = 3, max_offset: int = MAX_OFFSET) -> list:
    """Generate n random offsets for cropping."""
    offsets = []
    for _ in range(n):
        dx = random.randint(-max_offset, max_offset)
        dy = random.randint(-max_offset, max_offset)
        offsets.append((dx, dy))
    return offsets


def crop_with_offset(img: np.ndarray, offset: tuple, crop_size: int = CROP_SIZE) -> np.ndarray:
    """Crop image from center with given offset."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    dx, dy = offset

    x1 = cx + dx - crop_size // 2
    y1 = cy + dy - crop_size // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    return img[y1:y2, x1:x2].copy()


def add_text_overlay(img: np.ndarray, text: str = "DEFECT") -> np.ndarray:
    """Add thin bright green text to top-left corner."""
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    draw.text((10, 10), text, fill=(0, 255, 0), font=font)
    return np.array(pil_img)


def corrupt_channel(img: np.ndarray, method: str = "random") -> np.ndarray:
    """Corrupt image to make it very different from normal."""
    corrupted = img.copy()

    if method == "shuffle":
        np.random.shuffle(corrupted.reshape(-1, 3))
        corrupted = corrupted.reshape(img.shape)
    elif method == "noise":
        corrupted = np.random.randint(0, 256, img.shape, dtype=np.uint8)
    elif method == "solid":
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
        corrupted[:] = color
    elif method == "invert":
        corrupted = 255 - corrupted
    elif method == "shift":
        shift = np.random.randint(50, 150)
        corrupted = np.roll(corrupted, shift, axis=(0, 1))
    else:
        corrupted = np.random.randint(0, 256, img.shape, dtype=np.uint8)

    return corrupted


def generate_test_tif(
    img_path: Path,
    output_dir: Path,
    corruption_type: str = None,
    corrupt_idx: int = None,
    add_text: bool = False,
    text_idx: int = None,
) -> Path:
    """
    Generate a test .tif file with 3 slightly offset crops.

    Args:
        img_path: Source image path
        output_dir: Output directory
        corruption_type: Type of corruption ('noise', 'solid', 'invert', 'shift', None)
        corrupt_idx: Which image to corrupt (0, 1, or 2), None for no corruption
        add_text: Whether to add text overlay
        text_idx: Which image to add text to
    """
    img = np.array(Image.open(img_path).convert('RGB'))
    offsets = get_random_offsets(3)
    crops = [crop_with_offset(img, offset) for offset in offsets]

    if add_text and text_idx is not None:
        crops[text_idx] = add_text_overlay(crops[text_idx])

    if corruption_type and corrupt_idx is not None:
        crops[corrupt_idx] = corrupt_channel(crops[corrupt_idx], corruption_type)

    stacked = np.stack(crops, axis=0)

    suffix_parts = []
    if corruption_type and corrupt_idx is not None:
        suffix_parts.append(f"corrupt_{corruption_type}_{corrupt_idx}")
    if add_text and text_idx is not None:
        suffix_parts.append(f"text_{text_idx}")
    if not suffix_parts:
        suffix_parts.append("normal")

    suffix = "_".join(suffix_parts)
    output_name = f"{img_path.stem}_{suffix}.tif"
    output_path = output_dir / output_name

    tifffile.imwrite(output_path, stacked)
    return output_path


def main():
    source_dir = Path("/home/yclai/vscode_project/MLExploration_efficientAD/datasets/mvtec/carpet/train/good")
    output_dir = Path("/home/yclai/vscode_project/MLExploration_efficientAD/API_filter/broken_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_images = sorted(source_dir.glob("*.png"))[:10]

    corruption_types = ["noise", "solid", "invert", "shift"]

    generated = []

    for i, img_path in enumerate(source_images):
        print(f"Processing {img_path.name}...")

        if i < 2:
            out = generate_test_tif(img_path, output_dir)
            generated.append(("normal", out))

        elif i < 5:
            corrupt_type = corruption_types[(i - 2) % len(corruption_types)]
            corrupt_idx = random.randint(0, 2)
            out = generate_test_tif(
                img_path, output_dir,
                corruption_type=corrupt_type,
                corrupt_idx=corrupt_idx
            )
            generated.append((f"corrupt_{corrupt_type}", out))

        elif i < 8:
            text_idx = random.randint(0, 2)
            out = generate_test_tif(
                img_path, output_dir,
                add_text=True,
                text_idx=text_idx
            )
            generated.append(("text_only", out))

        else:
            corrupt_type = random.choice(corruption_types)
            corrupt_idx = random.randint(0, 2)
            text_idx = (corrupt_idx + 1) % 3
            out = generate_test_tif(
                img_path, output_dir,
                corruption_type=corrupt_type,
                corrupt_idx=corrupt_idx,
                add_text=True,
                text_idx=text_idx
            )
            generated.append(("corrupt_and_text", out))

    print(f"\n{'='*60}")
    print(f"Generated {len(generated)} test files:")
    print(f"{'='*60}")
    for category, path in generated:
        print(f"  [{category:20s}] {path.name}")

    print(f"\nOutput directory: {output_dir}")
    print(f"Expected shape per file: (3, {CROP_SIZE}, {CROP_SIZE}, 3)")


if __name__ == "__main__":
    main()
