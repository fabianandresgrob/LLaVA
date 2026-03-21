#!/bin/bash
#SBATCH --job-name=llava-sae-eo-pt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=08:00:00
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
OUTPUT_DIR="$SCRATCH/checkpoints/llava-v1.5-7b-pretrain-sae-encode-only"

# Verify pretrain data exists
if [ ! -f "$DATA_DIR/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json" ]; then
    echo "ERROR: Pretrain data not found at $DATA_DIR/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
    echo "Download with: huggingface-cli download liuhaotian/LLaVA-Pretrain --local-dir $DATA_DIR/LLaVA-Pretrain"
    exit 1
fi

echo "Python: $(which python) ($(python --version))"
nvidia-smi

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using MASTER_PORT=$MASTER_PORT"

# ---- Stage 1: Pretrain (feature alignment) ----
# Trains ONLY the new 8192->4096 projector. CLIP, LLM, and SAE are all frozen.
# conversation format: "plain" (no system prompt, simple image-caption pairs)
# ZeRO-2 is sufficient since only the projector is trainable.
# Global batch size = 128: per_device(8) x grad_accum(4) x gpus(4) = 128
deepspeed --master_port $MASTER_PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path "$DATA_DIR/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json" \
    --image_folder "$DATA_DIR/LLaVA-Pretrain/images" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    --sae_checkpoint_path "$SAE_CHECKPOINT_PATH"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date): Stage 1 pretrain completed successfully!"
    echo "Projector saved to: $OUTPUT_DIR"
    echo "Next: run slurm_finetune_sae_encode_only.sh"
else
    echo "$(date): Stage 1 pretrain CRASHED with exit code $EXIT_CODE. Check logs."
fi
