#!/bin/bash

# Multi-layer loss ablation study training script
# This trains EfficientAD with both conv5 and conv6 losses

echo "Starting multi-layer loss training..."
echo "Model will be saved to: ablation_studies/multi_layer_loss/results/"

cd /home/yclai/vscode_project/MLExploration_efficientAD

python ablation_studies/multi_layer_loss/src_run/train_multi_layer.py \
    --path ./datasets/VirtualSEM
