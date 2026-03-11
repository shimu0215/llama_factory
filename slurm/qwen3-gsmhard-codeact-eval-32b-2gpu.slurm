#!/bin/bash
#SBATCH --job-name=q3-gsm-32b
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --output=hpc-results/q3-gsm-32b-%j.out
#SBATCH --error=hpc-results/q3-gsm-32b-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=24:00:00

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

OUT_ROOT="/scratch/wzhao20/llama_factory/outputs/gsmhard_qwen3_scale_eval/32b_${SLURM_JOB_ID}"
INFER_YAML="examples/inference/qwen3_32b_codeact.yaml"
MODEL_32B="${MODEL_32B:-Qwen/Qwen3-32B}"

scripts/run_gsmhard_codeact_eval_one_model.sh "$MODEL_32B" qwen3_32b "$INFER_YAML" "$OUT_ROOT" 16384 0.85 0.0 5 1024 16384 2 1200
