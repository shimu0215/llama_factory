#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --time=12:00:00
#SBATCH --job-name=iter-a100-12h
#SBATCH --output=hpc-results/iter-a100-12h-%j.out
#SBATCH --error=hpc-results/iter-a100-12h-%j.err

set -euo pipefail

mkdir -p hpc-results

hostname
nvidia-smi
sleep 43200
