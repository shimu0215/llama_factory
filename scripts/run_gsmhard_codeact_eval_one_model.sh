#!/bin/bash
set -euo pipefail

MODEL_ID=${1:?usage: run_gsmhard_codeact_eval_one_model.sh <model_id> <model_tag> <infer_yaml> <output_root> [vllm_maxlen] [vllm_gpu_util] [temperature] [max_steps] [max_tokens] [prompt_budget_tokens] [recent_steps] [api_timeout]}
MODEL_TAG=${2:?missing model_tag}
INFER_YAML=${3:?missing infer_yaml}
OUTPUT_ROOT=${4:?missing output_root}
VLLM_MAXLEN=${5:-16384}
VLLM_GPU_UTIL=${6:-0.70}
TEMPERATURE=${7:-0.0}
MAX_STEPS=${8:-5}
MAX_TOKENS=${9:-1024}
PROMPT_BUDGET=${10:-16384}
RECENT_STEPS=${11:-2}
API_TIMEOUT=${12:-900}

WORKDIR=/scratch/wzhao20/llama_factory
OUTDIR="${OUTPUT_ROOT}/${MODEL_TAG}"
mkdir -p "$OUTDIR"

API_LOG="${OUTDIR}/api.log"
EVAL_LOG="${OUTDIR}/eval.log"

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
    if ! kill -0 "$API_PID" >/dev/null 2>&1; then
      echo "[ERROR] API process exited early for ${MODEL_TAG}" >&2
      tail -n 120 "$API_LOG" || true
      return 1
    fi
    if [ $(( $(date +%s) - start )) -ge "$timeout" ]; then
      echo "[ERROR] API readiness timeout (${timeout}s) for ${MODEL_TAG}" >&2
      tail -n 120 "$API_LOG" || true
      return 1
    fi
    sleep 2
  done
}

cleanup() {
  if [ -n "${API_PID:-}" ] && kill -0 "$API_PID" >/dev/null 2>&1; then
    kill -TERM -- "-$API_PID" >/dev/null 2>&1 || kill "$API_PID" >/dev/null 2>&1 || true
    sleep 3
    kill -KILL -- "-$API_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "$WORKDIR"
PORT=$(pick_port)

echo "[INFO] MODEL_ID=${MODEL_ID}" | tee "$EVAL_LOG"
echo "[INFO] MODEL_TAG=${MODEL_TAG}" | tee -a "$EVAL_LOG"
echo "[INFO] PORT=${PORT}" | tee -a "$EVAL_LOG"

env API_PORT="$PORT" \
  setsid llamafactory-cli api "$INFER_YAML" \
    model_name_or_path="$MODEL_ID" \
    infer_backend=vllm \
    vllm_enforce_eager=true \
    vllm_maxlen="$VLLM_MAXLEN" \
    vllm_gpu_util="$VLLM_GPU_UTIL" \
    >"$API_LOG" 2>&1 &
API_PID=$!

wait_api_ready "$PORT" "$API_TIMEOUT"

echo "[INFO] API ready, start eval for ${MODEL_TAG}" | tee -a "$EVAL_LOG"
python scripts/run_gsm_hard_smolagents_codeact.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --api-key 0 \
  --model test \
  --limit 0 \
  --num-samples 1 \
  --temperature "$TEMPERATURE" \
  --max-steps "$MAX_STEPS" \
  --max-tokens "$MAX_TOKENS" \
  --recent-steps "$RECENT_STEPS" \
  --prompt-budget-tokens "$PROMPT_BUDGET" \
  --record-shard-size 1000 \
  --group-shard-size 500 \
  --context-shard-size 1000 \
  --output-dir "$OUTDIR" \
  > >(tee -a "$EVAL_LOG") 2>&1

echo "[INFO] finished ${MODEL_TAG}" | tee -a "$EVAL_LOG"
