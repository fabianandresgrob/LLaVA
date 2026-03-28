#!/bin/bash
#SBATCH --job-name=llava-sae-eo-ft-13b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=1-12:00:00
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
if [ -z "$SCRATCH" ]; then
    echo "ERROR: \$SCRATCH is not set. Add it to your ~/.bashrc"
    exit 1
fi
if [ -z "$SAE_CHECKPOINT_PATH" ]; then
    echo "ERROR: \$SAE_CHECKPOINT_PATH is not set. Add it to your ~/.bashrc"
    exit 1
fi

DATA_DIR="$SCRATCH/llava_data"
PRETRAIN_PROJECTOR="$SCRATCH/checkpoints/llava-v1.5-13b-pretrain-sae-encode-only/mm_projector.bin"
OUTPUT_DIR="$SCRATCH/checkpoints/llava-v1.5-13b-finetune-sae-encode-only"

# Verify stage 1 projector exists
if [ ! -f "$PRETRAIN_PROJECTOR" ]; then
    echo "ERROR: Stage 1 projector not found at $PRETRAIN_PROJECTOR"
    echo "Run slurm_pretrain_sae_encode_only_13b.sh first."
    exit 1
fi

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

# ---- Stage 2: Finetune (visual instruction tuning) ----
# Trains projector + LLM. CLIP and SAE stay frozen.
# 4x H100 80GB, ZeRO-3.
# Global batch size = 128: per_device(2) x grad_accum(16) x gpus(4) = 128
deepspeed --master_port $MASTER_PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
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
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
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
    --sae_encode_only True \
    --sae_checkpoint_path "$SAE_CHECKPOINT_PATH" \
    --resume_from_checkpoint $RESUME_FLAG

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date): Stage 2 finetune completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
else
    echo "$(date): Stage 2 finetune CRASHED with exit code $EXIT_CODE. Check logs."
fi
