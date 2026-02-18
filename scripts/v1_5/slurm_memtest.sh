#!/bin/bash
#SBATCH --job-name=llava-memtest
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=00:15:00
#SBATCH --partition=mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
#SBATCH --qos=mcml

set -e
mkdir -p logs
source .venv/bin/activate

export CUDA_HOME="$HOME/cuda-12.1"
export HF_HUB_CACHE="$MCMLSCRATCH/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$MCMLSCRATCH/.cache/huggingface/datasets"

DATA_DIR="$MCMLSCRATCH/llava_data"
PRETRAIN_PROJECTOR="$MCMLSCRATCH/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin"
SAE_CHECKPOINT="$MCMLSCRATCH/checkpoints_dir/batch_top_k_20_x8/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_batch_top_k_20_x8/trainer_0/ae.pt"

echo "Python: $(which python) ($(python --version))"
nvidia-smi

# Train for 20 steps only — testing if batch_size=8 fits with CPU offload, and measuring step time
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
    --output_dir /tmp/llava-memtest \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --max_steps 20 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none \
    --use_sae_bottleneck True \
    --sae_checkpoint_path "$SAE_CHECKPOINT"

echo "Memory test passed! Peak GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
