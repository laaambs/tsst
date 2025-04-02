# #!/bin/bash
suffix="eval"
LOG_DIR=logs/$suffix
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/${suffix}_vcr_$(date +"%Y-%m-%d_%H-%M-%S").log

# CONFIG='config/eval/vcr_beam.yaml'
CONFIG='config/eval/vcr_direct.yaml'

export CUDA_VISIBLE_DEVICES=2

python -m evaluation.evaluation --yaml_path ${CONFIG} > $LOG_FILE 2>&1


