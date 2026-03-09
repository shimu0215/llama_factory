# Research Platform (smolagents + CodeAct + TRL)

This folder is a new, standalone experiment platform. It does **not** modify existing LlamaFactory workflows.

## Goals
- Unified CodeAct prompt and runtime for generation/eval and training data construction.
- Three experiment runners:
  1. `generate_eval_runner`: generate trajectories + evaluate first-sample accuracy.
  2. `kd_train_runner`: basic KD via trajectory SFT for student.
  3. `teacher_ft_runner`: teacher fine-tuning (current: SFT baseline; custom loss/RL hook ready in trainer).
- Stage-level resume checkpoints for interruption recovery.
- Context compression controls:
  - enable/disable toggle
  - fixed token budget or auto budget from model context limit

## Prompt consistency
All pipelines read a shared prompt file:
- `research_platform_trl/prompts/codeact_v1.yaml`

Keep this file fixed across data generation/training/eval to avoid prompt mismatch.

## Stage-level resume
Each runner writes stage checkpoint state under:
- `<work_dir>/checkpoints/*.json`

Run with `--resume` to skip completed stages.

## Runner 1: Generate + Eval
Config template:
- `research_platform_trl/configs/generate_eval_example.yaml`

Run:
```bash
cd /scratch/wzhao20/llama_factory
python -m research_platform_trl.runners.generate_eval_runner \
  --config research_platform_trl/configs/generate_eval_example.yaml
```

Resume:
```bash
python -m research_platform_trl.runners.generate_eval_runner \
  --config research_platform_trl/configs/generate_eval_example.yaml \
  --resume
```

Outputs:
- `<work_dir>/records.jsonl`
- `<work_dir>/contexts.jsonl`
- `<work_dir>/summary.json`

## Runner 2: KD Train (student)
Config template:
- `research_platform_trl/configs/kd_train_example.yaml`

Run:
```bash
python -m research_platform_trl.runners.kd_train_runner \
  --config research_platform_trl/configs/kd_train_example.yaml
```

Resume:
```bash
python -m research_platform_trl.runners.kd_train_runner \
  --config research_platform_trl/configs/kd_train_example.yaml \
  --resume
```

Outputs:
- `<work_dir>/train_data/kd_train.jsonl`
- `<work_dir>/student_model/`
- `<work_dir>/summary.json`

## Runner 3: Teacher FT (SFT baseline)
Config template:
- `research_platform_trl/configs/teacher_ft_example.yaml`

Run:
```bash
python -m research_platform_trl.runners.teacher_ft_runner \
  --config research_platform_trl/configs/teacher_ft_example.yaml
```

Resume:
```bash
python -m research_platform_trl.runners.teacher_ft_runner \
  --config research_platform_trl/configs/teacher_ft_example.yaml \
  --resume
```

Outputs:
- `<work_dir>/train_data/teacher_ft_train.jsonl`
- `<work_dir>/teacher_model/`
- `<work_dir>/summary.json`

## Notes
- Current KD/Teacher-FT uses TRL SFT (`SFTTrainer`) with chat messages.
- For future custom loss and RL, extend:
  - `research_platform_trl/trainers/teacher_sft.py`
