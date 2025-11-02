#!/bin/bash
set -e

# Train DRAEM using Anomalib official CLI
LOG_DIR="results"
mkdir -p "$LOG_DIR"

echo "Training DRAEM using Anomalib CLI..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomalib
anomalib train --config ./src_run/config_draem_cli.yaml
