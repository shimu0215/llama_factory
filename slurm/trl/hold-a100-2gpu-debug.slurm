#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --time=24:00:00
#SBATCH --job-name=hold-a100-2g
#SBATCH --output=hpc-results/hold-a100-2g-%j.out
#SBATCH --error=hpc-results/hold-a100-2g-%j.err

set -euo pipefail

mkdir -p hpc-results

hostname
nvidia-smi
sleep 86400
