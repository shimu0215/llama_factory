#!/bin/bash
set -euo pipefail

MODEL_ID=${1:?usage: run_trl_gsmhard_eval_model_aligned.sh <model_id> <model_tag> <infer_yaml>}
MODEL_TAG=${2:?missing model_tag}
INFER_YAML=${3:?missing infer_yaml}

RESULT_ROOT="${RESULT_ROOT:-/scratch/wzhao20/llama_factory/outputs/trl_results_aligned}"
RUN_DIR="${RUN_DIR:-$RESULT_ROOT/${MODEL_TAG}}"
DATA_DIR="$RUN_DIR/trl_generated_data"
CFG_PATH="$RUN_DIR/generate_eval.yaml"
API_LOG="$RUN_DIR/api.log"

NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_STEPS="${MAX_STEPS:-5}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MODEL_MAX_CONTEXT_TOKENS="${MODEL_MAX_CONTEXT_TOKENS:-16384}"
PROMPT_BUDGET_TOKENS="${PROMPT_BUDGET_TOKENS:-16384}"
VLLM_MAXLEN="${VLLM_MAXLEN:-16384}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.65}"
API_READY_TIMEOUT="${API_READY_TIMEOUT:-1200}"
FLUSH_EVERY_QUESTIONS="${FLUSH_EVERY_QUESTIONS:-1}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_LIMIT="${DATASET_LIMIT:-0}"
ENABLE_ROLLING_MEMORY_CODE_AGENT="${ENABLE_ROLLING_MEMORY_CODE_AGENT:-false}"
ENABLE_CONTEXT_COMPRESSION="${ENABLE_CONTEXT_COMPRESSION:-true}"

mkdir -p "$RESULT_ROOT" "$RUN_DIR" "$DATA_DIR"

pick_port() {
  python - <<'PY'
import socket
s=socket.socket()
s.bind(("127.0.0.1",0))
print(s.getsockname()[1])
s.close()
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
      echo "[ERROR] API process exited early." >&2
      tail -n 120 "$API_LOG" || true
      return 1
    fi
    if [ $(( $(date +%s) - start )) -ge "$timeout" ]; then
      echo "[ERROR] API readiness timeout (${timeout}s)." >&2
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

PORT=$(pick_port)
echo "[INFO] MODEL_ID=$MODEL_ID"
echo "[INFO] MODEL_TAG=$MODEL_TAG"
echo "[INFO] RUN_DIR=$RUN_DIR"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] API_PORT=$PORT"

env API_PORT="$PORT" \
  setsid llamafactory-cli api "$INFER_YAML" \
    model_name_or_path="$MODEL_ID" \
    infer_backend=vllm \
    vllm_enforce_eager=true \
    vllm_maxlen="$VLLM_MAXLEN" \
    vllm_gpu_util="$VLLM_GPU_UTIL" \
    >"$API_LOG" 2>&1 &
API_PID=$!

wait_api_ready "$PORT" "$API_READY_TIMEOUT"

cat > "$CFG_PATH" <<CFGEOF
model:
  backend: openai_compatible
  model_id: test
  api_base: http://127.0.0.1:${PORT}/v1
  api_key: "0"
  temperature: ${TEMPERATURE}
  max_tokens: ${MAX_TOKENS}

dataset:
  path: reasoning-machines/gsm-hard
  config: default
  split: ${DATASET_SPLIT}
  limit: ${DATASET_LIMIT}

tools:
  enabled:
    - python_exec

generation:
  num_samples: ${NUM_SAMPLES}
  max_steps: ${MAX_STEPS}
  temperature: ${TEMPERATURE}
  max_tokens: ${MAX_TOKENS}

context:
  enable_rolling_memory_code_agent: ${ENABLE_ROLLING_MEMORY_CODE_AGENT}
  enable_context_compression: ${ENABLE_CONTEXT_COMPRESSION}
  compression_mode: auto
  model_max_context_tokens: ${MODEL_MAX_CONTEXT_TOKENS}
  prompt_budget_tokens: ${PROMPT_BUDGET_TOKENS}
  recent_steps: 2
  max_summary_chars: 1200
  max_observation_chars: 1500
  max_step_chars: 1200

output:
  work_dir: ${DATA_DIR}
  flush_every_questions: ${FLUSH_EVERY_QUESTIONS}
CFGEOF

python -m research_platform_trl.runners.generate_eval_runner \
  --config "$CFG_PATH" \
  --resume
