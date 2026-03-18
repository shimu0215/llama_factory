#!/bin/bash
set -euo pipefail

# Run inside an existing Hopper allocation from /scratch/wzhao20/llama_factory.
# Uses AgentDistill 500-question sample and python-only, n=1, temp=0.

EVAL_SCRIPT="slurm/qwen-gsmhard-agentdistill-aligned-python-n1-temp0.slurm"
RESULT_BASE="${RESULT_BASE:-outputs/offline_agent_alignment}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${RESULT_BASE}_after_target2}"
mkdir -p "$RESULT_BASE"

THRESHOLD_05_MAX="${THRESHOLD_05_MAX:-0.10}"
THRESHOLD_15_MAX="${THRESHOLD_15_MAX:-0.15}"
THRESHOLD_7B_MIN="${THRESHOLD_7B_MIN:-0.50}"

read_acc() {
  local summary_path=$1
  python - <<'PY' "$summary_path"
import json,sys
p=sys.argv[1]
obj=json.load(open(p))
print(obj["metrics"]["first_sample_accuracy"])
PY
}

run_model() {
  local model_id=$1
  local tag=$2
  local out_dir="$RESULT_BASE/$tag"
  echo "[RUN] model=$model_id out_dir=$out_dir" >&2
  MODEL_ID="$model_id" MODEL_TAG="$tag" OUT_DIR="$out_dir" bash "$EVAL_SCRIPT" >&2
  local summary="$out_dir/gsm_hard_agentdistill_aligned_summary.json"
  if [[ ! -f "$summary" ]]; then
    echo "[ERROR] Missing summary file: $summary" >&2
    exit 1
  fi
  local acc
  acc=$(read_acc "$summary")
  echo "[METRIC] $tag first_sample_accuracy=$acc" >&2
  echo "$acc"
}

leq_threshold() {
  python - <<'PY' "$1" "$2"
import sys
x=float(sys.argv[1]); t=float(sys.argv[2])
print("1" if x <= t else "0")
PY
}

geq_threshold() {
  python - <<'PY' "$1" "$2"
import sys
x=float(sys.argv[1]); t=float(sys.argv[2])
print("1" if x >= t else "0")
PY
}

save_snapshot() {
  local src_dir=$1
  local dst_dir=$2
  rm -rf "$dst_dir"
  mkdir -p "$dst_dir"
  cp -R "$src_dir"/. "$dst_dir"/
  echo "[SNAPSHOT] saved to $dst_dir" >&2
}

# Target 1
acc_05=$(run_model "Qwen/Qwen2.5-0.5B-Instruct" "qwen25_0p5b")
if [[ "$(leq_threshold "$acc_05" "$THRESHOLD_05_MAX")" != "1" ]]; then
  echo "[STOP] Target 1 not met: 0.5B accuracy $acc_05 > $THRESHOLD_05_MAX"
  exit 2
fi

acc_7b_t1=$(run_model "Qwen/Qwen2.5-7B-Instruct" "qwen25_7b_after_t1")
if [[ "$(geq_threshold "$acc_7b_t1" "$THRESHOLD_7B_MIN")" != "1" ]]; then
  echo "[ALERT] 7B dropped below threshold after target 1: $acc_7b_t1 < $THRESHOLD_7B_MIN"
  exit 3
fi

# Target 2
acc_15=$(run_model "Qwen/Qwen2.5-1.5B-Instruct" "qwen25_1p5b")
if [[ "$(leq_threshold "$acc_15" "$THRESHOLD_15_MAX")" != "1" ]]; then
  echo "[STOP] Target 2 not met: 1.5B accuracy $acc_15 > $THRESHOLD_15_MAX"
  exit 4
fi

acc_7b_t2=$(run_model "Qwen/Qwen2.5-7B-Instruct" "qwen25_7b_after_t2")
if [[ "$(geq_threshold "$acc_7b_t2" "$THRESHOLD_7B_MIN")" != "1" ]]; then
  echo "[ALERT] 7B dropped below threshold after target 2: $acc_7b_t2 < $THRESHOLD_7B_MIN"
  exit 5
fi

save_snapshot "$RESULT_BASE" "$SNAPSHOT_DIR"

# Target 3 (optional): do not fail pipeline if missed.
acc_06=$(run_model "Qwen/Qwen3-0.6B" "qwen3_0p6b")
if [[ "$(leq_threshold "$acc_06" "$THRESHOLD_15_MAX")" == "1" ]]; then
  echo "[DONE] Target 3 met: 0.6B accuracy $acc_06 <= $THRESHOLD_15_MAX"
else
  echo "[DONE] Target 3 not met: 0.6B accuracy $acc_06 > $THRESHOLD_15_MAX"
fi

echo "[DONE] Sequence complete. Results under $RESULT_BASE"
