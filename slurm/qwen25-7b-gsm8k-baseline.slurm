#!/bin/bash
#SBATCH --job-name=lf-gsm8k-sft-7b
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.40gb:4
#SBATCH --output=lf-gsm8k-sft-7b-%j.out
#SBATCH --error=lf-gsm8k-sft-7b-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=24:00:00

set -euo pipefail

source ~/.bashrc
module load cuda/12.6

export HF_HOME=/scratch/wzhao20/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=/scratch/wzhao20/vllm_cache

cd /scratch/wzhao20/llama_factory
conda activate /home/wzhao20/miniconda3/envs/llama-factory311

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

export FORCE_TORCHRUN=1

llamafactory-cli train examples/train_lora/qwen25_7b_gsm8k_lora_sft_ds3.yaml
