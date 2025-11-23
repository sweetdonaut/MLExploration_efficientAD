#!/bin/bash
set -e

# Visualize standalone EfficientAD model with defect marking
LOG_DIR="results_standalone"
mkdir -p "$LOG_DIR"

echo "Testing standalone EfficientAD with defect marking..."
PYTHONPATH=$PWD:$PYTHONPATH python3 ./src_run/visualize_standalone_marking.py \
    --path ./datasets/VirtualSEM \
    --threshold 0.5 \
    --min-area 50
