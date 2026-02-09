#!/bin/bash
#SBATCH --job-name=llava-download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --partition=lrz-cpu

set -e
mkdir -p logs
bash scripts/download_llava_data.sh
