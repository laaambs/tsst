#!/bin/bash

# Configurable parameters
PROJECT_ROOT=""
VISION_MODEL_PATH=""
BASE_MODEL_PATH=""
OUTPUT_DIR=""
DATA_PATH=""
IMAGE_DIR=""

# NCCL settings
export NCCL_TIMEOUT=360  
export NCCL_IB_TIMEOUT=360

# Redirect output to log file
exec 1> >(tee -a ${OUTPUT_DIR}/training.log)
exec 2>&1

deepspeed --include localhost:2,3,4,5 --master_port 29506 llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --deepspeed ${PROJECT_ROOT}/models/LLaVA/scripts/zero3.json \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_DIR} \
    --vision_tower ${VISION_MODEL_PATH} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --freeze_mm_mlp_adapter True \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb
