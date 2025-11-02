#!/bin/bash
set -e

# Test Anomalib EfficientAD model (inference only)
LOG_DIR="results"
mkdir -p "$LOG_DIR"

echo "Testing Anomalib EfficientAD..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomalib
python3 test_original_anomalib.py 2>&1 | tee "$LOG_DIR/anomalib_testing_$(date +%Y%m%d_%H%M%S).log"
