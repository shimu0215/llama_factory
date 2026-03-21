#!/bin/bash
set -euo pipefail

set +u
source ~/.bashrc
set -u

source /home/wzhao20/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/wzhao20/conda_envs/llama-factory311-clean

cd /scratch/wzhao20/llama_factory

export HF_HOME="${HF_HOME:-/scratch/wzhao20/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wzhao20/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/wzhao20/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wzhao20/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wzhao20/torchinductor_cache}"

export RESULT_ROOT="${RESULT_ROOT:-/scratch/wzhao20/llama_factory/outputs/trl_results_accuracy_latest}"
export DATA_PATH="${DATA_PATH:-/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/gsm_hard_500_20250507.json}"
export DATASET_LIMIT="${DATASET_LIMIT:-500}"
export NUM_SAMPLES="${NUM_SAMPLES:-1}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export MAX_STEPS="${MAX_STEPS:-5}"
export MAX_TOKENS="${MAX_TOKENS:-1024}"
export MODEL_MAX_CONTEXT_TOKENS="${MODEL_MAX_CONTEXT_TOKENS:-16384}"
export PROMPT_BUDGET_TOKENS="${PROMPT_BUDGET_TOKENS:-16384}"
export VLLM_MAXLEN="${VLLM_MAXLEN:-16384}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.75}"
export FLUSH_EVERY_QUESTIONS="${FLUSH_EVERY_QUESTIONS:-1}"
export ENABLE_ROLLING_MEMORY_CODE_AGENT="${ENABLE_ROLLING_MEMORY_CODE_AGENT:-false}"
export ENABLE_CONTEXT_COMPRESSION="${ENABLE_CONTEXT_COMPRESSION:-false}"

mkdir -p "$RESULT_ROOT" /scratch/wzhao20/llama_factory/hpc-results

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct|gsmhard500_qwen25_0p5b_python_1sample_latesttrl_nocompress|examples/inference/qwen25_7b.yaml"
  "Qwen/Qwen2.5-1.5B-Instruct|gsmhard500_qwen25_1p5b_python_1sample_latesttrl_nocompress|examples/inference/qwen25_7b.yaml"
  "Qwen/Qwen3-0.6B|gsmhard500_qwen3_0p6b_python_1sample_latesttrl_nocompress|examples/inference/qwen3_4b_codeact.yaml"
  "Qwen/Qwen3-1.7B|gsmhard500_qwen3_1p7b_python_1sample_latesttrl_nocompress|examples/inference/qwen3_4b_codeact.yaml"
  "Qwen/Qwen3-4B|gsmhard500_qwen3_4b_python_1sample_latesttrl_nocompress|examples/inference/qwen3_4b_codeact.yaml"
  "Qwen/Qwen3-8B|gsmhard500_qwen3_8b_python_1sample_latesttrl_nocompress|examples/inference/qwen3_4b_codeact.yaml"
)

for spec in "${MODELS[@]}"; do
  IFS="|" read -r MODEL_ID MODEL_TAG INFER_YAML <<< "$spec"
  RUN_DIR="$RESULT_ROOT/$MODEL_TAG/trl_generated_data"
  SUMMARY_PATH="$RUN_DIR/summary.json"
  LOG_PATH="/scratch/wzhao20/llama_factory/hpc-results/${MODEL_TAG}.log"

  echo "[INFO] Starting $MODEL_TAG"
  echo "[INFO] model=$MODEL_ID yaml=$INFER_YAML"

  if [[ -f "$SUMMARY_PATH" ]]; then
    echo "[INFO] Found existing summary for $MODEL_TAG, skipping."
    continue
  fi

  bash slurm/trl/run_trl_gsmhard_eval_model_aligned.sh \
    "$MODEL_ID" \
    "$MODEL_TAG" \
    "$INFER_YAML" \
    2>&1 | tee "$LOG_PATH"
done
