#!/bin/bash
#SBATCH --job-name=qwen-aligned-py-t0
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=hpc-results/qwen-aligned-py-t0-%j.out
#SBATCH --error=hpc-results/qwen-aligned-py-t0-%j.err
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

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
INFER_YAML="${INFER_YAML:-examples/inference/qwen25_7b.yaml}"
# Use the exact 500-sample JSON used in AgentDistill paper runs.
DATA_PATH="${DATA_PATH:-/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/gsm_hard_500_20250507.json}"
MODEL_TAG="${MODEL_TAG:-$(echo "$MODEL_ID" | tr '/.' '__')}"
OUT_DIR="${OUT_DIR:-outputs/${MODEL_TAG}_gsmhard_aligned_python_n1_t0_${SLURM_JOB_ID}}"
API_TIMEOUT="${API_TIMEOUT:-1200}"
VLLM_MAXLEN="${VLLM_MAXLEN:-16384}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.65}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"

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
API_LOG="hpc-results/${MODEL_TAG}-aligned-api-${SLURM_JOB_ID}.log"

env API_PORT="$PORT" \
  setsid llamafactory-cli api "$INFER_YAML" \
    model_name_or_path="$MODEL_ID" \
    infer_backend=vllm \
    vllm_enforce_eager=true \
    vllm_maxlen="$VLLM_MAXLEN" \
    vllm_gpu_util="$VLLM_GPU_UTIL" \
    >"$API_LOG" 2>&1 &
APIPID=$!

wait_api_ready "$PORT" "$API_TIMEOUT"

python scripts/run_gsm_hard_agentdistill_aligned_eval.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --api-key 0 \
  --model test \
  --data-path "$DATA_PATH" \
  --limit "$EVAL_LIMIT" \
  --num-samples 1 \
  --temperature 0.0 \
  --max-steps 5 \
  --max-tokens 1024 \
  --tool-mode python_only \
  --output-dir "$OUT_DIR"

echo "[DONE] model=${MODEL_ID} output_dir=${OUT_DIR}"
