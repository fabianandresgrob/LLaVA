#!/bin/bash
#SBATCH --job-name=fix-scheduler
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:30:00

cd ~/LLaVA
source .venv/bin/activate

CHECKPOINT="$MCMLSCRATCH/checkpoints/llava-v1.5-7b-finetune-sae/checkpoint-3000"
python scripts/fix_checkpoint_scheduler.py "$CHECKPOINT"
