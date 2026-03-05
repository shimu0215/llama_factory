#!/bin/bash
#SBATCH --job-name=lf-qwen3-14b-thought-entropy-sft-full
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --output=hpc-results/lf-qwen3-14b-thought-entropy-sft-full-%j.out
#SBATCH --error=hpc-results/lf-qwen3-14b-thought-entropy-sft-full-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=36:00:00

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

RAW_DIR="/scratch/wzhao20/llama_factory/outputs/qwen3_14b_gsm_hard_smolagents_codeact"
TARGET_DATA="/scratch/wzhao20/llama_factory/data/gsm_hard_codeact_sft_thought_only_full.jsonl"
CKPT_STEPS="${CKPT_STEPS:-100}"

if ! ls "$RAW_DIR"/gsm_hard_codeact_records_part_*.jsonl >/dev/null 2>&1; then
  echo "No full records found at: $RAW_DIR"
  exit 1
fi

if [ ! -s "$TARGET_DATA" ] || ! grep -q '"role": "observation"' "$TARGET_DATA"; then
  echo "Building full thought-only SFT dataset from $RAW_DIR"
  python scripts/convert_codeact_to_thought_only_sft.py \
    --input-dir "$RAW_DIR" \
    --output "$TARGET_DATA" \
    --strict \
    --keep-think-tags
else
  echo "Found existing preprocessed dataset: $TARGET_DATA"
fi

unset RANK WORLD_SIZE LOCAL_RANK NODE_RANK MASTER_ADDR MASTER_PORT
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 1000))

IFNAME=$(ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
echo "Using interface: $IFNAME, MASTER_PORT=$MASTER_PORT, CKPT_STEPS=$CKPT_STEPS"

export GLOO_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_IFNAME=$IFNAME
export GLOO_USE_IPV6=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export FORCE_TORCHRUN=1

echo "Start full-data training at $(date)"
llamafactory-cli train examples/train_freeze/qwen3_14b_gsmhard_thought_only_entropy_sft_full.yaml \
  save_steps="$CKPT_STEPS"
echo "Finished at $(date)"
