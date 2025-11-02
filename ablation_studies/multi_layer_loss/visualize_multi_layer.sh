#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomalib

cd /home/yclai/vscode_project/MLExploration_efficientAD

python3 ablation_studies/multi_layer_loss/visualize_multi_layer.py --path ./datasets/VirtualSEM

echo ""
echo "Visualizations saved to:"
echo "  ablation_studies/multi_layer_loss/results/VirtualSEM/repeating_medium_multi_layer/pdn_layers/test/"
