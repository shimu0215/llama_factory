#!/bin/bash
#SBATCH --job-name=lf-qwen3-32b-codeact-gsmhard
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --output=hpc-results/lf-qwen3-32b-codeact-gsmhard-%j.out
#SBATCH --error=hpc-results/lf-qwen3-32b-codeact-gsmhard-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -eo pipefail
set +u
source ~/.bashrc
set -u
module load cuda/12.6

export HF_HOME=/scratch/wzhao20/hf_cache
export HF_DATASETS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=/scratch/wzhao20/vllm_cache
export TRITON_CACHE_DIR=/scratch/wzhao20/triton_cache

mkdir -p /scratch/wzhao20/triton_cache
mkdir -p /scratch/wzhao20/llama_factory/hpc-results
mkdir -p /scratch/wzhao20/llama_factory/outputs/qwen3_32b_gsm_hard_smolagents_codeact

cd /scratch/wzhao20/llama_factory
conda activate /scratch/wzhao20/conda_envs/llama-factory311-clean

unset RANK WORLD_SIZE LOCAL_RANK NODE_RANK MASTER_ADDR MASTER_PORT

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 1000))
export API_PORT=$((31000 + RANDOM % 1000))

IFNAME=$(ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
echo "Using interface: $IFNAME, MASTER_PORT=$MASTER_PORT, API_PORT=$API_PORT"

export GLOO_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_IFNAME=$IFNAME
export GLOO_USE_IPV6=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=16

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "$API_PID" || true
    wait "$API_PID" || true
  fi
}
trap cleanup EXIT

API_PORT=$API_PORT llamafactory-cli api examples/inference/qwen3_32b_codeact.yaml \
  vllm_enforce_eager=true \
  > hpc-results/lf-qwen3-32b-codeact-api-${SLURM_JOB_ID}.log 2>&1 &
API_PID=$!

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${API_PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:${API_PORT}/v1/models" >/dev/null

python scripts/run_gsm_hard_smolagents_codeact.py \
  --base-url "http://127.0.0.1:${API_PORT}/v1" \
  --model test \
  --limit 0 \
  --num-samples 3 \
  --temperature 0.7 \
  --max-steps 5 \
  --max-tokens 1024 \
  --recent-steps 2 \
  --prompt-budget-tokens 12000 \
  --max-summary-chars 1200 \
  --max-observation-chars 1500 \
  --max-step-chars 1200 \
  --record-shard-size 200 \
  --group-shard-size 50 \
  --output-dir outputs/qwen3_32b_gsm_hard_smolagents_codeact
