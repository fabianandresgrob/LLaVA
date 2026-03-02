#!/bin/bash
#SBATCH --job-name=llava-download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

# Requires $SCRATCH and $HF_HUB_CACHE to be set in ~/.bashrc:
#   export SCRATCH=/lustre/groups/eml/projects/<username>
#   export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"

set -e
mkdir -p logs

if [ -z "$SCRATCH" ]; then
    echo "ERROR: \$SCRATCH is not set. Add it to your ~/.bashrc"
    exit 1
fi

bash scripts/download_llava_data.sh
