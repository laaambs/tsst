# #!/bin/bash
suffix="eval"
LOG_DIR=logs/$suffix
mkdir -p $LOG_DIR

LOG_FILE=$LOG_DIR/${suffix}_$(date +"%Y-%m-%d_%H-%M-%S").log

# CONFIG='config/eval/scienceqa_beam.yaml'
CONFIG='config/eval/scienceqa_direct.yaml'

export CUDA_VISIBLE_DEVICES=4,5

python -m evaluation.evaluation --yaml_path ${CONFIG} > $LOG_FILE 2>&1