#!/bin/bash
#SBATCH --job-name=lf-qwen3-14b-thought-entropy-sft
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --output=hpc-results/lf-qwen3-14b-thought-entropy-sft-%j.out
#SBATCH --error=hpc-results/lf-qwen3-14b-thought-entropy-sft-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

TARGET_DATA="/scratch/wzhao20/llama_factory/data/gsm_hard_codeact_sft_thought_only.jsonl"
if [ ! -s "$TARGET_DATA" ]; then
  echo "Preprocessed dataset not found at $TARGET_DATA, running preprocessing..."
  CANDIDATE_DIRS=(
    "/scratch/wzhao20/llama_factory/outputs/qwen3_14b_gsm_hard_smolagents_codeact"
    "/scratch/wzhao20/llama_factory/outputs/qwen3_14b_gsm_hard_smolagents_codeact_dev_50q"
    "/scratch/wzhao20/llama_factory/outputs/qwen3_14b_gsm_hard_smolagents_codeact_preview_5q"
    "/scratch/wzhao20/llama_factory/outputs/qwen3_32b_gsm_hard_smolagents_codeact"
  )
  INPUT_DIR=""
  for d in "${CANDIDATE_DIRS[@]}"; do
    if ls "$d"/gsm_hard_codeact_records*.jsonl >/dev/null 2>&1; then
      INPUT_DIR="$d"
      break
    fi
  done

  if [ -z "$INPUT_DIR" ]; then
    echo "No raw codeact records found in candidate output dirs."
    exit 1
  fi

  echo "Using raw records from: $INPUT_DIR"
  python scripts/convert_codeact_to_thought_only_sft.py \
    --input-dir "$INPUT_DIR" \
    --output "$TARGET_DATA" \
    --strict \
    --keep-think-tags
  echo "Preprocessing finished: $TARGET_DATA"
else
  echo "Found preprocessed dataset: $TARGET_DATA"
fi

unset RANK WORLD_SIZE LOCAL_RANK NODE_RANK MASTER_ADDR MASTER_PORT

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 1000))

IFNAME=$(ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
echo "Using interface: $IFNAME, MASTER_PORT=$MASTER_PORT"

export GLOO_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_IFNAME=$IFNAME
export GLOO_USE_IPV6=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export FORCE_TORCHRUN=1

echo "Start training at $(date)"
llamafactory-cli train examples/train_freeze/qwen3_14b_gsmhard_thought_only_entropy_sft.yaml
echo "Finished at $(date)"
