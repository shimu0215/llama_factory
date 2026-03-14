#!/bin/bash
#SBATCH --job-name=trl-gsm-q3-1p7b
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=hpc-results/trl-gsm-q3-1p7b-%j.out
#SBATCH --error=hpc-results/trl-gsm-q3-1p7b-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=48:00:00

set -eo pipefail
set +u
source ~/.bashrc
set -u
module load cuda/12.6

export HF_HOME=/scratch/wzhao20/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=/scratch/wzhao20/vllm_cache
export TRITON_CACHE_DIR=/scratch/wzhao20/triton_cache
mkdir -p "$TRITON_CACHE_DIR" hpc-results

cd /scratch/wzhao20/llama_factory
conda activate /scratch/wzhao20/conda_envs/llama-factory311-clean

export NUM_SAMPLES=1
export MAX_STEPS=5
export MODEL_MAX_CONTEXT_TOKENS=16384
export PROMPT_BUDGET_TOKENS=16384
export FLUSH_EVERY_QUESTIONS=5

bash slurm/trl/run_trl_gsmhard_eval_model.sh \
  "Qwen/Qwen3-1.7B" \
  "gsmhard_qwen3_1p7b_python_1sample_no_rolling" \
  "examples/inference/qwen3_4b_codeact.yaml"

