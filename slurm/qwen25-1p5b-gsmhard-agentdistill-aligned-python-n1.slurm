#!/bin/bash
#SBATCH --job-name=q25-1p5b-aligned-py
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=hpc-results/q25-1p5b-aligned-py-%j.out
#SBATCH --error=hpc-results/q25-1p5b-aligned-py-%j.err
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

MODEL_15B="${MODEL_15B:-Qwen/Qwen2.5-1.5B-Instruct}"
INFER_YAML="examples/inference/qwen25_7b.yaml"
OUT_DIR="${OUT_DIR:-outputs/qwen25_1p5b_gsmhard_aligned_n1_python_${SLURM_JOB_ID}}"
API_TIMEOUT="${API_TIMEOUT:-900}"

pick_port() {
  python - <<'PY'
import socket
s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()
PY
}

wait_api_ready() {
  local port=$1
  local timeout=$2
  local start
  start=$(date +%s)
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$APIPID" >/dev/null 2>&1; then
      echo "[ERROR] API process exited early" >&2
      tail -n 120 "$API_LOG" || true
      return 1
    fi
    if [ $(( $(date +%s) - start )) -ge "$timeout" ]; then
      echo "[ERROR] API readiness timeout (${timeout}s)" >&2
      tail -n 120 "$API_LOG" || true
      return 1
    fi
    sleep 2
  done
}

cleanup() {
  if [ -n "${APIPID:-}" ] && kill -0 "$APIPID" >/dev/null 2>&1; then
    kill -TERM -- "-$APIPID" >/dev/null 2>&1 || kill "$APIPID" >/dev/null 2>&1 || true
    sleep 3
    kill -KILL -- "-$APIPID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

PORT=$(pick_port)
API_LOG="hpc-results/q25-1p5b-aligned-py-api-${SLURM_JOB_ID}.log"

env API_PORT="$PORT" \
  setsid llamafactory-cli api "$INFER_YAML" \
    model_name_or_path="$MODEL_15B" \
    infer_backend=vllm \
    vllm_enforce_eager=true \
    vllm_maxlen=16384 \
    vllm_gpu_util=0.65 \
    >"$API_LOG" 2>&1 &
APIPID=$!

wait_api_ready "$PORT" "$API_TIMEOUT"

python scripts/run_gsm_hard_agentdistill_aligned_eval.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --api-key 0 \
  --model test \
  --dataset-name reasoning-machines/gsm-hard \
  --dataset-config default \
  --split train \
  --limit 0 \
  --num-samples 1 \
  --temperature 0.4 \
  --max-steps 5 \
  --max-tokens 1024 \
  --tool-mode python_only \
  --output-dir "$OUT_DIR"

echo "[DONE] output dir: $OUT_DIR"
