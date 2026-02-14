#!/bin/bash
#SBATCH --job-name=llava-sae-ft
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --partition=mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --signal=B:SIGUSR1@300

set -e

mkdir -p logs

# ---- Activate environment ----
source .venv/bin/activate

# ---- Storage paths ----
export CUDA_HOME="$HOME/cuda-12.1"
export HF_HUB_CACHE="$MCMLSCRATCH/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$MCMLSCRATCH/.cache/huggingface/datasets"

DATA_DIR="$MCMLSCRATCH/llava_data"
PRETRAIN_PROJECTOR="$MCMLSCRATCH/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin"
SAE_CHECKPOINT="$MCMLSCRATCH/checkpoints_dir/batch_top_k_20_x8/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_batch_top_k_20_x8/trainer_0/ae.pt"
OUTPUT_DIR="$MCMLSCRATCH/checkpoints/llava-v1.5-7b-finetune-sae"

# ---- Handle preemption/timeout: save checkpoint on SIGUSR1 ----
# SLURM sends SIGUSR1 300s before timeout (--signal=B:SIGUSR1@300)
# The HF Trainer catches SIGTERM and saves a checkpoint before exiting
handle_signal() {
    echo "$(date): Received signal, letting trainer save checkpoint..."
    # Send SIGTERM to the deepspeed process so Trainer saves checkpoint
    kill -TERM $(jobs -p) 2>/dev/null
    wait
    echo "$(date): Trainer exited after signal. Resubmitting job..."
    sbatch "$0"
    exit 0
}
trap handle_signal SIGUSR1

# ---- Resume logic ----
# The training script (train.py:978-981) already detects checkpoint-* dirs
# in output_dir and passes resume_from_checkpoint=True automatically.
# The SAE bottleneck is re-initialized fresh from SAE_CHECKPOINT every time
# (it's frozen and excluded from training checkpoints by design).
if [ -d "$OUTPUT_DIR" ] && ls "$OUTPUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
    echo "$(date): Checkpoint(s) found in $OUTPUT_DIR — will auto-resume"
else
    echo "$(date): No checkpoint found — starting fresh training"
fi

echo "Python: $(which python) ($(python --version))"
nvidia-smi

# ---- Training ----
# Save every 500 steps (~12% of epoch). Keeps last 3 checkpoints.
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
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
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
    --sae_checkpoint_path "$SAE_CHECKPOINT" &

# Wait for training process (needed for signal handling)
wait $!
EXIT_CODE=$?

# Only resubmit if training was actually making progress but ran out of time.
# The signal handler (SIGUSR1) handles the timeout case and resubmits there.
# If we get here with exit code 0, training completed successfully.
# If non-zero, it's a real crash — don't resubmit.
if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date): Training completed successfully!"
else
    echo "$(date): Training CRASHED with exit code $EXIT_CODE. Check logs. NOT resubmitting."
fi
