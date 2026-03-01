#!/bin/bash
#SBATCH --job-name=llava-sae-ft
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --partition=mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --signal=B:SIGUSR1@900
#SBATCH --no-requeue

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
# SLURM sends SIGUSR1 900s before timeout (--signal=B:SIGUSR1@900).
# We send SIGTERM directly to the Python worker (not the deepspeed launcher),
# so the HF Trainer's SIGTERM handler fires: sets should_save=True +
# should_training_stop=True, saves after the current step, then exits cleanly.
# The launcher then detects the worker exited and shuts down too.
handle_signal() {
    echo "$(date): Received signal, sending SIGTERM to training worker for graceful checkpoint save..."
    # Deepspeed spawns a 3-level process tree: runner (LAUNCHER_PID) -> launch.py -> training worker.
    # All three have train_mem.py in their command line. Exclude the runner (LAUNCHER_PID) and
    # take the highest PID — Linux assigns PIDs sequentially, so the worker (spawned last) has the highest.
    # Verified with 3-level simulation test (test_signal_3level.sh) on the login node.
    echo "$(date): All train_mem.py PIDs: $(pgrep -f train_mem.py | tr '\n' ' ') | LAUNCHER_PID=$LAUNCHER_PID"
    WORKER_PID=$(pgrep -f train_mem.py | grep -v "^${LAUNCHER_PID}$" | sort -n | tail -1)
    echo "$(date): Targeting worker PID: $WORKER_PID"
    if [ -n "$WORKER_PID" ]; then
        kill -TERM "$WORKER_PID" 2>/dev/null
    else
        echo "$(date): Worker PID not found, falling back to killing launcher"
        kill -TERM "$LAUNCHER_PID" 2>/dev/null
    fi
    # 'wait' inside a trap handler can return early in bash — poll instead
    while kill -0 "$LAUNCHER_PID" 2>/dev/null; do sleep 1; done
    echo "$(date): Trainer exited. Resubmitting job..."
    sbatch "$0"
    exit 0
}
trap handle_signal SIGUSR1

# ---- Resume logic ----
# The training script already detects checkpoint-* dirs
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

# Pick a free port to avoid collisions if two jobs land on the same node
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using MASTER_PORT=$MASTER_PORT"

# ---- Training ----
# Save every 500 steps (~12% of epoch). Keeps last 3 checkpoints.
deepspeed --master_port $MASTER_PORT llava/train/train_mem.py \
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
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_sae_bottleneck True \
    --sae_checkpoint_path "$SAE_CHECKPOINT" &
LAUNCHER_PID=$!

# Wait for training process (needed for signal handling)
wait $LAUNCHER_PID
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
