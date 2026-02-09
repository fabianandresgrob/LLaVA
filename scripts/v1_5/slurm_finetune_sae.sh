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

# ---- Storage paths ----
# Redirect HuggingFace cache to scratch (vicuna-7b ~14GB, CLIP ~1.7GB auto-download here)
export HF_HUB_CACHE="$MCMLSCRATCH/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$MCMLSCRATCH/.cache/huggingface/datasets"

# Where the training data lives (images + annotation JSON)
# You need to download and organize under this directory:
#   $DATA_DIR/llava_v1_5_mix665k.json   (annotation, ~1GB)
#   $DATA_DIR/coco/train2017/            (COCO images, ~19GB)
#   $DATA_DIR/gqa/images/                (GQA images, ~20GB)
#   $DATA_DIR/ocr_vqa/images/            (OCR-VQA images as .jpg, ~33GB)
#   $DATA_DIR/textvqa/train_images/      (TextVQA images, ~7GB)
#   $DATA_DIR/vg/VG_100K/               (VisualGenome part1, ~15GB)
#   $DATA_DIR/vg/VG_100K_2/             (VisualGenome part2)
DATA_DIR="$MCMLSCRATCH/llava_data"

# Stage 1 pretrained projector (download once from HuggingFace):
#   huggingface-cli download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5
# Then point to the mm_projector.bin file:
PRETRAIN_PROJECTOR="$MCMLSCRATCH/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin"

# SAE checkpoint (already trained)
SAE_CHECKPOINT="$MCMLSCRATCH/checkpoints_dir/batch_top_k_20_x8/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_batch_top_k_20_x8/trainer_0/ae.pt"

echo "Python: $(which python) ($(python --version))"
nvidia-smi

# ---- Single GPU, ZeRO-3 + CPU offload ----
# Effective batch size: 4 * 32 * 1 GPU = 128
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
    --output_dir "$MCMLSCRATCH/checkpoints/llava-v1.5-7b-finetune-sae" \
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
