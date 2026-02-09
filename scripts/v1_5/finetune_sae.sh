#!/bin/bash

# LLaVA 1.5 7B instruction tuning WITH SAE bottleneck
# SAE (BatchTopK, K=20, x8) inserted between CLIP vision encoder and projection MLP
#
# Single GPU (80GB H100/A100): uses ZeRO-3 + CPU offload
#   Effective batch size: 4 * 32 * 1 GPU = 128
#
# For multi GPU (4x), change to:
#   --deepspeed ./scripts/zero3.json
#   --per_device_train_batch_size 16
#   --gradient_accumulation_steps 2
#
# ---- Configure these paths ----
DATA_DIR="./playground/data"
PRETRAIN_PROJECTOR="./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin"
SAE_CHECKPOINT="../sae-for-vlm/checkpoints_dir/batch_top_k_20_x8/"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path "$DATA_DIR/llava_v1_5_mix665k.json" \
    --image_folder "$DATA_DIR" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter "$PRETRAIN_PROJECTOR" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-finetune-sae \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_sae_bottleneck True \
    --sae_checkpoint_path "$SAE_CHECKPOINT"
