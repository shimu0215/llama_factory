#!/bin/bash
#SBATCH --job-name=lf-gsm8k-smoke-2gpu
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --output=hpc-results/lf-gsm8k-smoke-2gpu-%j.out
#SBATCH --error=hpc-results/lf-gsm8k-smoke-2gpu-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

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
conda activate /scratch/wzhao20/conda_envs/llama-factory311

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

# Preferred smoke path: no deepspeed
llamafactory-cli train examples/train_lora/qwen25_7b_gsm8k_lora_sft_smoke_2gpu.yaml

# Candidate ZeRO3 test (uncomment after smoke passes)
# llamafactory-cli train examples/train_lora/qwen25_7b_gsm8k_lora_sft_ds3_2gpu_candidate.yaml
