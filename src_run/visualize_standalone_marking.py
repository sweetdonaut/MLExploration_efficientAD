#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from torch.utils.data import DataLoader
from tqdm import tqdm

from efficientad.models.torch_model import EfficientAdModel, EfficientAdModelSize, reduce_tensor_elems


def rgb_loader(path):
    img = Image.open(path)
    if img.mode == 'L':
        img = img.convert('RGB')
    return img


def create_anomalib_jet_colormap():
    colormap_values = np.array([
        [0, 0, 143], [0, 0, 255], [0, 127, 255], [0, 255, 255],
        [127, 255, 127], [255, 255, 0], [255, 127, 0], [255, 0, 0], [127, 0, 0]
    ], dtype=np.float32)

    x = np.linspace(0, len(colormap_values) - 1, len(colormap_values))
    x_new = np.linspace(0, len(colormap_values) - 1, 256)

    r = np.interp(x_new, x, colormap_values[:, 0])
    g = np.interp(x_new, x, colormap_values[:, 1])
    b = np.interp(x_new, x, colormap_values[:, 2])

    return np.stack([r, g, b], axis=1).astype(np.uint8)


def detect_defects(anomaly_map, threshold=0.5, min_area=50):
    binary_mask = (anomaly_map > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return valid_contours


def draw_defect_markings(image, contours):
    mask_img = np.zeros_like(image)

    for contour in contours:
        cv2.drawContours(mask_img, [contour], -1, (255, 255, 255), -1)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(mask_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return mask_img


def create_visualization(image_path, map_st, map_stae, anomaly_map, gt_mask_path=None, vis_size=(256, 256), threshold=0.5, min_area=50):
    img = Image.open(image_path).convert('RGB').resize(vis_size)
    img_np = np.array(img)

    panel1 = img_np.copy()

    if gt_mask_path and gt_mask_path.exists():
        gt_mask = np.array(Image.open(gt_mask_path).convert('L').resize(vis_size))
        panel2 = np.zeros((vis_size[0], vis_size[1], 3), dtype=np.uint8)
        panel2[gt_mask > 0] = [255, 255, 255]
    else:
        panel2 = np.zeros((vis_size[0], vis_size[1], 3), dtype=np.uint8)

    jet_colormap = create_anomalib_jet_colormap()

    map_st_uint8 = (np.clip(map_st, 0, 1) * 255).astype(np.uint8)
    heatmap_st = jet_colormap[map_st_uint8]
    panel3 = cv2.addWeighted(img_np, 0.5, heatmap_st, 0.5, 0)

    map_stae_uint8 = (np.clip(map_stae, 0, 1) * 255).astype(np.uint8)
    heatmap_stae = jet_colormap[map_stae_uint8]
    panel4 = cv2.addWeighted(img_np, 0.5, heatmap_stae, 0.5, 0)

    anomaly_map_uint8 = (np.clip(anomaly_map, 0, 1) * 255).astype(np.uint8)
    heatmap_final = jet_colormap[anomaly_map_uint8]
    panel5 = cv2.addWeighted(img_np, 0.5, heatmap_final, 0.5, 0)

    defect_contours = detect_defects(anomaly_map, threshold=threshold, min_area=min_area)
    panel6 = draw_defect_markings(img_np, defect_contours)

    text_height = 40
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    def add_text_above_panel(panel, text):
        text_area = np.ones((text_height, panel.shape[1], 3), dtype=np.uint8) * 255
        text_img = Image.fromarray(text_area)
        draw = ImageDraw.Draw(text_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        x_pos = (panel.shape[1] - (bbox[2] - bbox[0])) // 2
        y_pos = (text_height - (bbox[3] - bbox[1])) // 2
        draw.text((x_pos, y_pos), text, font=font, fill=(0, 0, 0))
        return np.vstack([np.array(text_img), panel])

    return np.hstack([
        add_text_above_panel(panel1, 'Image'),
        add_text_above_panel(panel2, 'GT Mask'),
        add_text_above_panel(panel3, 'Image + map_st'),
        add_text_above_panel(panel4, 'Image + map_stae'),
        add_text_above_panel(panel5, 'Image + Final'),
        add_text_above_panel(panel6, 'Defect Marking')
    ])


parser = argparse.ArgumentParser(description='Visualize EfficientAD model with defect marking')
parser.add_argument('--path', type=str, default='./datasets/VirtaulSEM', help='Path to dataset root directory')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for defect detection (0-1)')
parser.add_argument('--min-area', type=int, default=50, help='Minimum area for valid defect regions (pixels)')
args = parser.parse_args()

category = "repeating"
model_size = EfficientAdModelSize.M

if model_size == EfficientAdModelSize.DINO:
    model_size_suffix = "_dinov3"
    image_size = (896, 896)
elif model_size == EfficientAdModelSize.M:
    model_size_suffix = "_medium"
    image_size = (256, 256)
else:
    model_size_suffix = "_small"
    image_size = (256, 256)

data_root = Path(args.path)
dataset_name = data_root.name
model_path = Path(f"results_standalone/EfficientAd/{dataset_name}/{category}{model_size_suffix}/model.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

test_path = data_root / category / "test"
test_transform = Compose([Resize(image_size), ToImage(), ToDtype(torch.float32, scale=True)])
test_dataset = ImageFolder(test_path, transform=test_transform, loader=rgb_loader)
good_idx = test_dataset.class_to_idx['good']

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = EfficientAdModel(teacher_out_channels=384, model_size=model_size, pad_maps=True)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

state_dict = checkpoint['state_dict']
state_dict_no_prefix = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict_no_prefix)
model = model.to(device).eval()

maps_st, maps_ae = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Computing quantiles"):
        for img, label in zip(images, labels):
            if label == good_idx:
                map_st, map_ae = model.get_maps(img.unsqueeze(0).to(device), normalize=False)
                maps_st.append(map_st)
                maps_ae.append(map_ae)

maps_st_flat = reduce_tensor_elems(torch.cat(maps_st))
maps_ae_flat = reduce_tensor_elems(torch.cat(maps_ae))

model.quantiles.qa_st.data = torch.quantile(maps_st_flat, q=0.9).to(device).data
model.quantiles.qb_st.data = torch.quantile(maps_st_flat, q=0.995).to(device).data
model.quantiles.qa_ae.data = torch.quantile(maps_ae_flat, q=0.9).to(device).data
model.quantiles.qb_ae.data = torch.quantile(maps_ae_flat, q=0.995).to(device).data

output_dir = Path("results_standalone/EfficientAd") / dataset_name / f"{category}{model_size_suffix}" / "images/test"
output_dir.mkdir(parents=True, exist_ok=True)

mask_root = data_root / category / "ground_truth"

with torch.no_grad():
    for idx, (images, labels) in enumerate(tqdm(test_loader, desc="Visualizing")):
        img_path, label = test_dataset.samples[idx]
        img_path = Path(img_path)
        defect_type = img_path.parent.name

        map_st, map_stae = model.get_maps(images.to(device), normalize=True)
        anomaly_map = 0.5 * map_st + 0.5 * map_stae

        print(f"{img_path.name}: map_st [{map_st.min():.4f}, {map_st.max():.4f}], map_stae [{map_stae.min():.4f}, {map_stae.max():.4f}]")

        gt_mask_path = mask_root / defect_type / f"{img_path.stem}_mask.png" if label != good_idx else None

        vis_image = create_visualization(
            img_path,
            map_st.cpu().squeeze().numpy(),
            map_stae.cpu().squeeze().numpy(),
            anomaly_map.cpu().squeeze().numpy(),
            gt_mask_path,
            vis_size=image_size,
            threshold=args.threshold,
            min_area=args.min_area
        )

        defect_output_dir = output_dir / defect_type
        defect_output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(defect_output_dir / f"{img_path.stem}.png"), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

print(f"\nVisualizations saved to: {output_dir}")
