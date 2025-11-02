#!/bin/bash
set -e

# Train standalone EfficientAD model
LOG_DIR="results_standalone"
mkdir -p "$LOG_DIR"

echo "Training standalone EfficientAD..."
PYTHONPATH=$PWD:$PYTHONPATH python3 ./src_run/train_standalone.py --path ./datasets/VirtualSEM 
