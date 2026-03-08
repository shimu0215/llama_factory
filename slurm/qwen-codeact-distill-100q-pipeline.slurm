#!/bin/bash
#SBATCH --job-name=lf-codeact-distill-100q
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=hpc-results/lf-codeact-distill-100q-%j.out
#SBATCH --error=hpc-results/lf-codeact-distill-100q-%j.err
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

SOURCE_DIR="/scratch/wzhao20/llama_factory/outputs/qwen3_14b_gsm_hard_smolagents_codeact"
WORK_DIR="/scratch/wzhao20/llama_factory/outputs/distill_100q_experiment_${SLURM_JOB_ID}"
FT14_PATH="${FT14_PATH:-/scratch/wzhao20/llama_factory/outputs/gsmhard_thought_only_entropy_sft_full}"

echo "SOURCE_DIR=$SOURCE_DIR"
echo "WORK_DIR=$WORK_DIR"
echo "FT14_PATH=$FT14_PATH"

python scripts/run_codeact_distill_100q_experiment.py \
  --source-records-dir "$SOURCE_DIR" \
  --finetuned-qwen14-path "$FT14_PATH" \
  --work-dir "$WORK_DIR" \
  --sample-size 100 \
  --seed 42 \
  --prompt-budget-tokens 16384 \
  --vllm-maxlen 24576 \
  --max-steps 5 \
  --max-tokens 1024 \
  --temperature 0.7 \
  --api-ready-timeout 1200 \
  --student-epochs 1.0 \
  --student-save-steps 50 \
  --student-grad-acc 32 \
  --student-lr 1e-4
