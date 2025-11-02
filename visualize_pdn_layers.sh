#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomalib

PYTHONPATH=$PWD:$PYTHONPATH python3 ./src_run/visualize_pdn_layers.py --path ./datasets/VirtualSEM
