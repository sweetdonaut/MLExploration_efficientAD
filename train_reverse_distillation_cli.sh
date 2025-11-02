#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomalib

PYTHONPATH=$PWD:$PYTHONPATH anomalib train \
    --config ./src_run/config_reverse_distillation_cli.yaml
