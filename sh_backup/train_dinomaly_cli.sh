#!/bin/bash
set -e

# Train Dinomaly using Anomalib official CLI
LOG_DIR="results"
mkdir -p "$LOG_DIR"

echo "Training Dinomaly using Anomalib CLI..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomalib
anomalib train --config ./src_run/config_dinomaly_cli.yaml
