#!/bin/bash
set -e

# Train SuperSimpleNet using Anomalib official CLI
LOG_DIR="results"
mkdir -p "$LOG_DIR"

echo "Training SuperSimpleNet using Anomalib CLI..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomalib
anomalib train --config ./src_run/config_supersimplenet_cli.yaml
