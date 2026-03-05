#!/bin/bash
#SBATCH --job-name=llava-sae-ft
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

set -e

mkdir -p logs

# ---- Activate environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava

# ---- CUDA: match the version PyTorch was compiled with to avoid DeepSpeed JIT mismatch ----
TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
echo "PyTorch compiled with CUDA: $TORCH_CUDA_VER"
if [ -d "/usr/local/cuda-$TORCH_CUDA_VER" ]; then
    export CUDA_HOME="/usr/local/cuda-$TORCH_CUDA_VER"
else
    export CUDA_HOME="/usr/local/cuda"
fi
echo "Using CUDA_HOME=$CUDA_HOME"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ---- Storage paths ----
# Set these in ~/.bashrc — do not hardcode personal paths in this script:
#   export SCRATCH=<your project directory>
#   export SAE_CHECKPOINT_PATH=<path to ae.pt>
#   export HF_HUB_CACHE=$SCRATCH/.cache/huggingface/hub
if [ -z "$SCRATCH" ]; then
    echo "ERROR: \$SCRATCH is not set. Add it to your ~/.bashrc"
    exit 1
fi
if [ -z "$SAE_CHECKPOINT_PATH" ]; then
    echo "ERROR: \$SAE_CHECKPOINT_PATH is not set. Add it to your ~/.bashrc"
    exit 1
fi

DATA_DIR="$SCRATCH/llava_data"
PRETRAIN_PROJECTOR="$SCRATCH/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin"
OUTPUT_DIR="$SCRATCH/checkpoints/llava-v1.5-7b-finetune-sae"

echo "Python: $(which python) ($(python --version))"
nvidia-smi

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using MASTER_PORT=$MASTER_PORT"

# ---- Resume logic ----
if [ -d "$OUTPUT_DIR" ] && ls "$OUTPUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
    echo "$(date): Checkpoint(s) found in $OUTPUT_DIR — will auto-resume"
    RESUME_FLAG="True"
else
    echo "$(date): No checkpoint found — starting fresh training"
    RESUME_FLAG="False"
fi

# ---- Training ----
# 4x H100 80GB, ZeRO-2 (no CPU offload needed).
# Global batch size = 128: per_device(4) x grad_accum(8) x gpus(4) = 128
deepspeed --master_port $MASTER_PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
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
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
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
    --sae_checkpoint_path "$SAE_CHECKPOINT_PATH" \
    --resume_from_checkpoint $RESUME_FLAG

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date): Training completed successfully!"
else
    echo "$(date): Training CRASHED with exit code $EXIT_CODE. Check logs."
fi
