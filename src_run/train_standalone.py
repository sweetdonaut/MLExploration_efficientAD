#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from torch.utils.data import DataLoader
from PIL import Image

from efficientad.trainer import EfficientADTrainer
from efficientad.models.torch_model import EfficientAdModelSize

def rgb_loader(path):
    img = Image.open(path)
    if img.mode == 'L':
        img = img.convert('RGB')
    return img

parser = argparse.ArgumentParser(description='Train EfficientAD model')
parser.add_argument('--path', type=str, default='./datasets/VirtaulSEM', help='Path to dataset root directory')
args = parser.parse_args()

category = "repeating"
data_root = Path(args.path)
imagenet_dir = Path("./datasets/imagenette")
max_epochs = 70
batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

train_path = data_root / category / "train"
test_path = data_root / category / "test"

if not train_path.exists() or not test_path.exists():
    print(f"Error: Data not found")
    sys.exit(1)

model_size = EfficientAdModelSize.M

train_transform = Compose([Resize((256, 256)), ToImage(), ToDtype(torch.float32, scale=True)])
train_dataset = ImageFolder(train_path, transform=train_transform, loader=rgb_loader)
test_dataset = ImageFolder(test_path, transform=train_transform, loader=rgb_loader)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=device == "cuda",
)

val_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=device == "cuda",
)

trainer = EfficientADTrainer(
    model_size=model_size,
    teacher_out_channels=384,
    imagenet_dir=str(imagenet_dir),
    lr=0.0001, 
    weight_decay=0.00001,
    device=device,
)

trainer.fit(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    max_epochs=max_epochs,
)

if model_size == EfficientAdModelSize.DINO:
    model_size_suffix = "_dinov3"
elif model_size == EfficientAdModelSize.M:
    model_size_suffix = "_medium"
else:
    model_size_suffix = "_small"

dataset_name = data_root.name
save_path = Path("results_standalone") / "EfficientAd" / dataset_name / f"{category}{model_size_suffix}" / "model.pth"
save_path.parent.mkdir(parents=True, exist_ok=True)
trainer.save_model(save_path)
print(f"Model saved to: {save_path}")
