#!/bin/bash
set -e

# Test standalone EfficientAD model
LOG_DIR="results_standalone"
mkdir -p "$LOG_DIR"

echo "Testing standalone EfficientAD..."
PYTHONPATH=$PWD:$PYTHONPATH python3 ./src_run/visualize_standalone.py --path ./datasets/VirtualSEM 
