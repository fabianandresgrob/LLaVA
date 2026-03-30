#!/bin/bash
#SBATCH --job-name=llava-eval-13b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

set -e

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava

TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
if [ -d "/usr/local/cuda-$TORCH_CUDA_VER" ]; then
    export CUDA_HOME="/usr/local/cuda-$TORCH_CUDA_VER"
else
    export CUDA_HOME="/usr/local/cuda"
fi
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if [ -z "$SCRATCH" ]; then
    echo "ERROR: \$SCRATCH is not set."
    exit 1
fi

SAE_MODEL_DIR="$SCRATCH/checkpoints/llava-v1.5-13b-finetune-sae"
SAE_ENCODE_ONLY_DIR="$SCRATCH/checkpoints/llava-v1.5-13b-finetune-sae-encode-only"
RESULTS_DIR="$SCRATCH/results/sae_llava_models_13b"
LMMS_EVAL_DIR="$HOME/projects/lmms-eval"

mkdir -p "$RESULTS_DIR"

echo "$(date): Running baseline model (liuhaotian/llava-v1.5-13b)"
cd "$LMMS_EVAL_DIR"
python -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-13b \
    --tasks vlms_are_biased,vilp,vlind_bench \
    --batch_size 1 \
    --output_path "$RESULTS_DIR/baseline" \
    --log_samples \
    --verbosity INFO

echo "$(date): Running SAE model ($SAE_MODEL_DIR)"
python -m lmms_eval \
    --model llava \
    --model_args pretrained="$SAE_MODEL_DIR" \
    --tasks vlms_are_biased,vilp,vlind_bench \
    --batch_size 1 \
    --output_path "$RESULTS_DIR/sae" \
    --log_samples \
    --verbosity INFO

echo "$(date): Running SAE encode-only model ($SAE_ENCODE_ONLY_DIR)"
python -m lmms_eval \
    --model llava \
    --model_args pretrained="$SAE_ENCODE_ONLY_DIR" \
    --tasks vlms_are_biased,vilp,vlind_bench \
    --batch_size 1 \
    --output_path "$RESULTS_DIR/sae_encode_only" \
    --log_samples \
    --verbosity INFO

echo "$(date): Evaluation complete. Results in $RESULTS_DIR"
