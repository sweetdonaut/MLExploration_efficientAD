#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

# Add paths to find modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from ablation_studies.multi_layer_loss.models.torch_model import EfficientAdModel, EfficientAdModelSize, imagenet_norm_batch


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


class PDNLayerExtractor:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.layer_mean_std = {}

    def compute_all_layers_mean_std(self, dataloader):
        """Compute mean/std for all teacher layers using training data."""
        print("Computing mean/std for all teacher layers...")
        teacher = self.model.teacher

        layer_stats = {}
        layer_names = [
            'after_conv1_relu', 'after_pool1',
            'after_conv2_relu', 'after_pool2',
            'after_conv3_relu', 'before_conv4_relu',
            'before_conv5_relu', 'after_conv6'
        ]

        for layer_name in layer_names:
            layer_stats[layer_name] = {
                'n': None,
                'sum': None,
                'sum_sqr': None
            }

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Computing statistics"):
                images = images.to(self.model.teacher.conv1.weight.device)
                x_t = imagenet_norm_batch(images)

                x_t = F.relu(teacher.conv1(x_t))
                self._update_stats(layer_stats['after_conv1_relu'], x_t)
                x_t = teacher.avgpool1(x_t)
                self._update_stats(layer_stats['after_pool1'], x_t)

                x_t = F.relu(teacher.conv2(x_t))
                self._update_stats(layer_stats['after_conv2_relu'], x_t)
                x_t = teacher.avgpool2(x_t)
                self._update_stats(layer_stats['after_pool2'], x_t)

                x_t = F.relu(teacher.conv3(x_t))
                self._update_stats(layer_stats['after_conv3_relu'], x_t)
                x_t_conv4_before_relu = teacher.conv4(x_t)
                self._update_stats(layer_stats['before_conv4_relu'], x_t_conv4_before_relu)
                x_t = F.relu(x_t_conv4_before_relu)
                x_t_conv5_before_relu = teacher.conv5(x_t)
                self._update_stats(layer_stats['before_conv5_relu'], x_t_conv5_before_relu)
                x_t = F.relu(x_t_conv5_before_relu)
                x_t = teacher.conv6(x_t)
                self._update_stats(layer_stats['after_conv6'], x_t)

        for layer_name, stats in layer_stats.items():
            mean = stats['sum'] / stats['n']
            std = torch.sqrt((stats['sum_sqr'] / stats['n']) - (mean ** 2))
            self.layer_mean_std[layer_name] = {
                'mean': mean[None, :, None, None],
                'std': std[None, :, None, None]
            }

        print("✓ All layer statistics computed")

    def _update_stats(self, stats, tensor):
        """Update running statistics for a layer."""
        if stats['n'] is None:
            _, num_channels, _, _ = tensor.shape
            stats['n'] = torch.zeros((num_channels,), dtype=torch.int64, device=tensor.device)
            stats['sum'] = torch.zeros((num_channels,), dtype=torch.float32, device=tensor.device)
            stats['sum_sqr'] = torch.zeros((num_channels,), dtype=torch.float32, device=tensor.device)

        stats['n'] += tensor[:, 0].numel()
        stats['sum'] += torch.sum(tensor, dim=[0, 2, 3])
        stats['sum_sqr'] += torch.sum(tensor ** 2, dim=[0, 2, 3])

    def extract_all_layers(self, x):
        teacher = self.model.teacher
        student = self.model.student
        self.activations = {}

        with torch.no_grad():
            x_t = imagenet_norm_batch(x)

            x_t = F.relu(teacher.conv1(x_t))
            self.activations['teacher_after_conv1_relu'] = x_t.detach()
            x_t = teacher.avgpool1(x_t)
            self.activations['teacher_after_pool1'] = x_t.detach()

            x_t = F.relu(teacher.conv2(x_t))
            self.activations['teacher_after_conv2_relu'] = x_t.detach()
            x_t = teacher.avgpool2(x_t)
            self.activations['teacher_after_pool2'] = x_t.detach()

            x_t = F.relu(teacher.conv3(x_t))
            self.activations['teacher_after_conv3_relu'] = x_t.detach()
            x_t_conv4_before_relu = teacher.conv4(x_t)
            self.activations['teacher_before_conv4_relu'] = x_t_conv4_before_relu.detach()
            x_t = F.relu(x_t_conv4_before_relu)
            x_t_conv5_before_relu = teacher.conv5(x_t)
            self.activations['teacher_before_conv5_relu'] = x_t_conv5_before_relu.detach()
            x_t = F.relu(x_t_conv5_before_relu)
            x_t = teacher.conv6(x_t)
            self.activations['teacher_after_conv6'] = x_t.detach()

        x_s = imagenet_norm_batch(x)
        x_s = F.relu(student.conv1(x_s))
        self.activations['student_after_conv1_relu'] = x_s.detach()
        x_s = student.avgpool1(x_s)
        self.activations['student_after_pool1'] = x_s.detach()

        x_s = F.relu(student.conv2(x_s))
        self.activations['student_after_conv2_relu'] = x_s.detach()
        x_s = student.avgpool2(x_s)
        self.activations['student_after_pool2'] = x_s.detach()

        x_s = F.relu(student.conv3(x_s))
        self.activations['student_after_conv3_relu'] = x_s.detach()
        x_s_conv4_before_relu = student.conv4(x_s)
        self.activations['student_before_conv4_relu'] = x_s_conv4_before_relu.detach()
        x_s = F.relu(x_s_conv4_before_relu)
        x_s_conv5_before_relu = student.conv5(x_s)
        self.activations['student_before_conv5_relu'] = x_s_conv5_before_relu.detach()
        x_s = F.relu(x_s_conv5_before_relu)
        x_s = student.conv6(x_s)
        self.activations['student_after_conv6'] = x_s.detach()

        return x_s


def compute_layer_heatmap(student_layer, teacher_layer, layer_name, layer_mean_std, image_size=(256, 256)):
    """Compute heatmap using different methods for different layers.

    - Conv4 (before ReLU): Use L2 distance without normalization (same as training)
    - Conv5 (before ReLU): Use L2 distance without normalization (same as training)
    - Conv6: Use L2 distance with normalization (same as training)
    - Other layers: Use L2 distance without normalization (for visualization only)
    """
    # Trim student channels if needed
    if student_layer.shape[1] > teacher_layer.shape[1]:
        student_layer = student_layer[:, :teacher_layer.shape[1], :, :]
    elif teacher_layer.shape[1] < student_layer.shape[1]:
        student_layer = student_layer[:, :teacher_layer.shape[1], :, :]

    # Conv4 (before ReLU): Use L2 distance without normalization
    if layer_name == 'before_conv4_relu':
        distance = torch.pow(teacher_layer - student_layer, 2)
        heatmap = torch.mean(distance, dim=1, keepdim=True)

    # Conv5 (before ReLU): Use L2 distance without normalization
    elif layer_name == 'before_conv5_relu':
        distance = torch.pow(teacher_layer - student_layer, 2)
        heatmap = torch.mean(distance, dim=1, keepdim=True)

    # Conv6: Use L2 distance with normalization (EfficientAD method)
    elif layer_name == 'after_conv6':
        if layer_name in layer_mean_std:
            mean = layer_mean_std[layer_name]['mean']
            std = layer_mean_std[layer_name]['std']
            teacher_layer = (teacher_layer - mean) / std

        distance = torch.pow(teacher_layer - student_layer, 2)
        heatmap = torch.mean(distance, dim=1, keepdim=True)

    # Other layers: Use L2 distance without normalization (for visualization)
    else:
        distance = torch.pow(teacher_layer - student_layer, 2)
        heatmap = torch.mean(distance, dim=1, keepdim=True)

    # Resize to image size if needed
    if heatmap.shape[-2:] != image_size:
        heatmap = F.interpolate(heatmap, size=image_size, mode='bilinear')

    return heatmap


def create_layer_visualization(image_path, layer_heatmaps, layer_names, vis_size=(256, 256)):
    jet_colormap = create_anomalib_jet_colormap()

    img = Image.open(image_path).convert('RGB').resize(vis_size)
    img_np = np.array(img)

    panels = [('Original', img_np, None, None)]

    rf_sizes = {
        'after_conv1_relu': '4x4',
        'after_pool1': '5x5',
        'after_conv2_relu': '11x11',
        'after_pool2': '13x13',
        'after_conv3_relu': '13x13',
        'before_conv4_relu': '21x21 (L2,pre-ReLU)',
        'before_conv5_relu': '33x33 (L2,pre-ReLU)',
        'after_conv6': '33x33 (L2)'
    }

    for layer_name in layer_names:
        if layer_name not in layer_heatmaps:
            continue

        heatmap_raw = layer_heatmaps[layer_name]
        vmin = heatmap_raw.min()
        vmax = heatmap_raw.max()

        if vmax > vmin:
            heatmap_normalized = (heatmap_raw - vmin) / (vmax - vmin)
        else:
            heatmap_normalized = np.zeros_like(heatmap_raw)

        heatmap_uint8 = (np.clip(heatmap_normalized, 0, 1) * 255).astype(np.uint8)
        heatmap_colored = jet_colormap[heatmap_uint8]

        display_name = layer_name.replace('after_', '').replace('_relu', '')
        label = f"{display_name}\nRF:{rf_sizes.get(layer_name, '?')}"

        panels.append((label, heatmap_colored, vmin, vmax))

    text_height = 40
    colorbar_height = 30

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    def add_header_to_panel(panel, text, vmin, vmax):
        panel_width = panel.shape[1]

        text_area = np.ones((text_height, panel_width, 3), dtype=np.uint8) * 255
        text_img = Image.fromarray(text_area)
        draw = ImageDraw.Draw(text_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        x_pos = (panel_width - (bbox[2] - bbox[0])) // 2
        y_pos = (text_height - (bbox[3] - bbox[1])) // 2
        draw.text((x_pos, y_pos), text, font=font, fill=(0, 0, 0))

        colorbar_area = np.ones((colorbar_height, panel_width, 3), dtype=np.uint8) * 255

        if vmin is not None and vmax is not None:
            bar_width = panel_width - 40
            bar_height = 12
            bar_x = 20
            bar_y = 2

            gradient = np.linspace(0, 255, bar_width).astype(np.uint8)
            gradient_colored = jet_colormap[gradient]
            colorbar_area[bar_y:bar_y+bar_height, bar_x:bar_x+bar_width] = gradient_colored

            colorbar_img = Image.fromarray(colorbar_area)
            draw_bar = ImageDraw.Draw(colorbar_img)
            draw_bar.text((bar_x, bar_y + bar_height + 1), f"{vmin:.2e}", font=font_small, fill=(0, 0, 0))
            draw_bar.text((bar_x + bar_width - 30, bar_y + bar_height + 1), f"{vmax:.2e}", font=font_small, fill=(0, 0, 0))
            colorbar_area = np.array(colorbar_img)

        header = np.vstack([np.array(text_img), colorbar_area])
        return np.vstack([header, panel])

    labeled_panels = [add_header_to_panel(panel, label, vmin, vmax) for label, panel, vmin, vmax in panels]
    return np.hstack(labeled_panels)


parser = argparse.ArgumentParser(description='Visualize PDN layer activations')
parser.add_argument('--path', type=str, default='./datasets/VirtualSEM')
args = parser.parse_args()

category = "repeating"
model_size = EfficientAdModelSize.M
model_size_suffix = "_medium" if model_size == EfficientAdModelSize.M else "_small"
image_size = (256, 256)

data_root = Path(args.path)
dataset_name = data_root.name
model_path = Path(f"ablation_studies/multi_layer_loss/results/{dataset_name}/{category}{model_size_suffix}_multi_layer/model.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

train_path = data_root / category / "train"
test_path = data_root / category / "test"
test_transform = Compose([Resize(image_size), ToImage(), ToDtype(torch.float32, scale=True)])
train_dataset = ImageFolder(train_path, transform=test_transform, loader=rgb_loader)
test_dataset = ImageFolder(test_path, transform=test_transform, loader=rgb_loader)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = EfficientAdModel(teacher_out_channels=384, model_size=model_size, pad_maps=True)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
state_dict = checkpoint['state_dict']
state_dict_no_prefix = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict_no_prefix)
model = model.to(device).eval()

extractor = PDNLayerExtractor(model)
extractor.compute_all_layers_mean_std(train_loader)

layer_names = [
    'after_conv1_relu',
    'after_pool1',
    'after_conv2_relu',
    'after_pool2',
    'after_conv3_relu',
    'before_conv4_relu',
    'before_conv5_relu',
    'after_conv6'
]

output_dir = Path("ablation_studies/multi_layer_loss/results") / dataset_name / f"{category}{model_size_suffix}_multi_layer" / "pdn_layers/test"
output_dir.mkdir(parents=True, exist_ok=True)

print("Visualizing PDN layers...")
with torch.no_grad():
    for idx, (images, labels) in enumerate(tqdm(test_loader, desc="Visualizing")):
        img_path, label = test_dataset.samples[idx]
        img_path = Path(img_path)
        defect_type = img_path.parent.name

        img_batch = images.to(device)
        _ = extractor.extract_all_layers(img_batch)

        layer_heatmaps = {}
        for layer_name in layer_names:
            student_key = f'student_{layer_name}'
            teacher_key = f'teacher_{layer_name}'

            if student_key in extractor.activations and teacher_key in extractor.activations:
                heatmap = compute_layer_heatmap(
                    extractor.activations[student_key],
                    extractor.activations[teacher_key],
                    layer_name,
                    extractor.layer_mean_std,
                    image_size=image_size
                )
                heatmap_np = heatmap.cpu().squeeze().numpy()


                layer_heatmaps[layer_name] = heatmap_np

        vis_image = create_layer_visualization(img_path, layer_heatmaps, layer_names, vis_size=image_size)

        defect_output_dir = output_dir / defect_type
        defect_output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(defect_output_dir / f"{img_path.stem}.png"),
            cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        )

print(f"\nPDN layer visualizations saved to: {output_dir}")
