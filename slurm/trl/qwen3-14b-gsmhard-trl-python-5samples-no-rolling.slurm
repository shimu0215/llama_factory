#!/bin/bash
#SBATCH --job-name=trl-gsmhard-14b-5s
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=hpc-results/trl-gsmhard-14b-5s-%j.out
#SBATCH --error=hpc-results/trl-gsmhard-14b-5s-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=48:00:00

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

RESULT_ROOT="${RESULT_ROOT:-/scratch/wzhao20/llama_factory/outputs/trl_results}"
RUN_DIR="${RUN_DIR:-$RESULT_ROOT/gsmhard_qwen3_14b_python_5samples_no_rolling}"
DATA_DIR="$RUN_DIR/trl_generated_data"
CFG_PATH="$RUN_DIR/generate_eval.yaml"
API_LOG="$RUN_DIR/api.log"

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
  local timeout=${2:-1200}
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
echo "[INFO] RESULT_ROOT=$RESULT_ROOT"
echo "[INFO] RUN_DIR=$RUN_DIR"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] API_PORT=$PORT"

env API_PORT="$PORT" \
  setsid llamafactory-cli api examples/inference/qwen3_14b_codeact.yaml \
    model_name_or_path="Qwen/Qwen3-14B" \
    infer_backend=vllm \
    vllm_enforce_eager=true \
    vllm_maxlen=16384 \
    vllm_gpu_util=0.65 \
    >"$API_LOG" 2>&1 &
API_PID=$!

wait_api_ready "$PORT" 1200

cat > "$CFG_PATH" <<EOF
model:
  backend: openai_compatible
  model_id: test
  api_base: http://127.0.0.1:${PORT}/v1
  api_key: "0"
  temperature: 0.7
  max_tokens: 1024

dataset:
  path: reasoning-machines/gsm-hard
  config: default
  split: test
  limit: 0

tools:
  enabled:
    - python_exec

generation:
  num_samples: 5
  max_steps: 5
  temperature: 0.7
  max_tokens: 1024

context:
  enable_rolling_memory_code_agent: false
  enable_context_compression: true
  compression_mode: auto
  model_max_context_tokens: 16384
  prompt_budget_tokens: 16384
  recent_steps: 2
  max_summary_chars: 1200
  max_observation_chars: 1500
  max_step_chars: 1200

output:
  work_dir: ${DATA_DIR}
EOF

python -m research_platform_trl.runners.generate_eval_runner \
  --config "$CFG_PATH" \
  --resume

