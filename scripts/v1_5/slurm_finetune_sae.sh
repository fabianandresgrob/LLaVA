#!/bin/bash
#SBATCH --job-name=llava-sae-ft
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml

set -e

mkdir -p logs

# ---- Activate environment ----
# Adjust to match your LLaVA environment (conda or venv)
conda activate llava

# ---- Cache dirs (avoid filling home quota) ----
export HF_HUB_CACHE="$MCMLSCRATCH/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$MCMLSCRATCH/.cache/huggingface/datasets"

echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "GPU Check:"
nvidia-smi

# ---- Single GPU, ZeRO-3 + CPU offload ----
# Effective batch size: 4 * 32 * 1 GPU = 128
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
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
    --sae_checkpoint_path ../sae-for-vlm/checkpoints_dir/batch_top_k_20_x8/
