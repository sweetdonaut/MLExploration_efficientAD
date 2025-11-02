#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from efficientad.models.torch_model import EfficientAdModel, EfficientAdModelSize, reduce_tensor_elems


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


def create_visualization(image_path, anomaly_map, pred_score, gt_mask_path=None, threshold=0.5):
    img = Image.open(image_path).convert('RGB').resize((256, 256))
    img_np = np.array(img)

    panel1 = img_np.copy()

    if gt_mask_path and gt_mask_path.exists():
        gt_mask = np.array(Image.open(gt_mask_path).convert('L').resize((256, 256)))
        panel2 = np.zeros((256, 256, 3), dtype=np.uint8)
        panel2[gt_mask > 0] = [255, 255, 255]
    else:
        panel2 = np.zeros((256, 256, 3), dtype=np.uint8)

    anomaly_map_uint8 = (np.clip(anomaly_map, 0, 1) * 255).astype(np.uint8)
    jet_colormap = create_anomalib_jet_colormap()
    heatmap = jet_colormap[anomaly_map_uint8]
    panel3 = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    panel4 = img_np.copy()
    pred_mask = (anomaly_map > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        cv2.drawContours(panel4, [contour], -1, (255, 0, 0), 2)
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                if ellipse[1][0] > 0 and ellipse[1][1] > 0:
                    cv2.ellipse(panel4, ellipse, (255, 0, 0), 2)
            except:
                pass

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
        add_text_above_panel(panel2, 'Gt Mask'),
        add_text_above_panel(panel3, 'Image + Anomaly Map'),
        add_text_above_panel(panel4, 'Image + Pred Mask')
    ])

parser = argparse.ArgumentParser(description='Test EfficientAD model')
parser.add_argument('--path', type=str, default='./datasets/mvtec', help='Path to dataset root directory')
args = parser.parse_args()

category = "grid"
data_root = Path(args.path)
dataset_name = data_root.name
model_path = Path(f"results_standalone/EfficientAd/{dataset_name}/{category}/model.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

test_path = data_root / category / "test"
test_transform = Compose([Resize((256, 256)), ToImage(), ToDtype(torch.float32, scale=True)])
test_dataset = ImageFolder(test_path, transform=test_transform)
good_idx = test_dataset.class_to_idx['good']

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = EfficientAdModel(teacher_out_channels=384, model_size=EfficientAdModelSize.S)
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
                map_st, map_ae = model.get_maps(img.to(device), normalize=False)
                maps_st.append(map_st)
                maps_ae.append(map_ae)

maps_st_flat = reduce_tensor_elems(torch.cat(maps_st))
maps_ae_flat = reduce_tensor_elems(torch.cat(maps_ae))

model.quantiles.qa_st.data = torch.quantile(maps_st_flat, q=0.9).to(device).data
model.quantiles.qb_st.data = torch.quantile(maps_st_flat, q=0.995).to(device).data
model.quantiles.qa_ae.data = torch.quantile(maps_ae_flat, q=0.9).to(device).data
model.quantiles.qb_ae.data = torch.quantile(maps_ae_flat, q=0.995).to(device).data

all_pred_scores, all_gt_labels, all_anomaly_maps, all_image_info = [], [], [], []

with torch.no_grad():
    for idx, (images, labels) in enumerate(tqdm(test_loader, desc="Inference")):
        output = model(images.to(device))
        img_path, label = test_dataset.samples[idx]
        img_path = Path(img_path)

        all_pred_scores.append(output.pred_score.cpu().item())
        all_gt_labels.append(0 if labels[0] == good_idx else 1)
        all_anomaly_maps.append(output.anomaly_map.cpu().squeeze())
        all_image_info.append((img_path, label, img_path.parent.name))

all_pred_scores = np.array(all_pred_scores)
all_gt_labels = np.array(all_gt_labels)

image_auroc = roc_auc_score(all_gt_labels, all_pred_scores)
image_f1 = f1_score(all_gt_labels, (all_pred_scores >= 0.5).astype(int))

mask_root = data_root / category / "ground_truth"
pixel_auroc, pixel_f1 = None, None

if mask_root.exists():
    first_img = Image.open(test_dataset.samples[0][0])
    original_h, original_w = first_img.size[1], first_img.size[0]

    all_pixel_scores, all_pixel_labels = [], []
    for i, (img_path, label) in enumerate(test_dataset.samples):
        img_path = Path(img_path)
        defect_type = img_path.parent.name

        anomaly_map_upsampled = torch.nn.functional.interpolate(
            all_anomaly_maps[i].unsqueeze(0).unsqueeze(0),
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        if label == good_idx:
            mask = np.zeros((original_h, original_w), dtype=bool)
        else:
            mask_path = mask_root / defect_type / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                continue
            mask = (read_image(str(mask_path), mode=ImageReadMode.GRAY).squeeze().numpy() > 0)

        all_pixel_scores.extend(anomaly_map_upsampled.flatten())
        all_pixel_labels.extend(mask.flatten().astype(int))

    if len(all_pixel_labels) > 0:
        all_pixel_scores = np.array(all_pixel_scores)
        all_pixel_labels = np.array(all_pixel_labels)
        pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
        pixel_f1 = f1_score(all_pixel_labels, (all_pixel_scores >= 0.5).astype(int))

print(f"\nResults:")
print(f"  Image AUROC: {image_auroc:.4f}, F1: {image_f1:.4f}")
if pixel_auroc:
    print(f"  Pixel AUROC: {pixel_auroc:.4f}, F1: {pixel_f1:.4f}")

output_dir = Path("results_standalone/EfficientAd") / dataset_name / category / "images/test"
output_dir.mkdir(parents=True, exist_ok=True)

for idx, (img_path, label, defect_type) in enumerate(tqdm(all_image_info, desc="Visualizations")):
    gt_mask_path = mask_root / defect_type / f"{img_path.stem}_mask.png" if label != good_idx else None
    vis_image = create_visualization(img_path, all_anomaly_maps[idx].numpy(), all_pred_scores[idx], gt_mask_path, 0.5)

    defect_output_dir = output_dir / defect_type
    defect_output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(defect_output_dir / f"{img_path.stem}.png"), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

print(f"\nVisualizations saved to: {output_dir}")
