#!/bin/bash
#SBATCH --job-name=trl-rejudge-q3-0p6b
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --output=hpc-results/trl-rejudge-q3-0p6b-%j.out
#SBATCH --error=hpc-results/trl-rejudge-q3-0p6b-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -eo pipefail
set +u
source ~/.bashrc
set -u

cd /scratch/wzhao20/llama_factory
conda activate /scratch/wzhao20/conda_envs/llama-factory311-clean
mkdir -p hpc-results

python scripts/recompute_trl_gsmhard_agentdistill_accuracy.py \
  --input-dir /scratch/wzhao20/llama_factory/outputs/trl_results_aligned/gsmhard_train_qwen3_0p6b_python_1sample_aligned_no_rolling/trl_generated_data
