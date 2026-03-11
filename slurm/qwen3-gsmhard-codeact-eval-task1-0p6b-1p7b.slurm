#!/bin/bash
#SBATCH --job-name=q3-gsmhard-t1
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=hpc-results/q3-gsmhard-t1-%j.out
#SBATCH --error=hpc-results/q3-gsmhard-t1-%j.err
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

OUT_ROOT="/scratch/wzhao20/llama_factory/outputs/gsmhard_qwen3_scale_eval/task1_${SLURM_JOB_ID}"
INFER_YAML="examples/inference/qwen3_4b_codeact.yaml"

MODEL_06B="${MODEL_06B:-Qwen/Qwen3-0.6B}"
MODEL_17B="${MODEL_17B:-Qwen/Qwen3-1.7B}"

scripts/run_gsmhard_codeact_eval_one_model.sh "$MODEL_06B" qwen3_0p6b "$INFER_YAML" "$OUT_ROOT" 16384 0.65 0.0 5 1024 16384 2 900
scripts/run_gsmhard_codeact_eval_one_model.sh "$MODEL_17B" qwen3_1p7b "$INFER_YAML" "$OUT_ROOT" 16384 0.65 0.0 5 1024 16384 2 900

echo "[DONE] outputs at $OUT_ROOT"
