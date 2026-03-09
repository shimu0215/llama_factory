import argparse
import json
from pathlib import Path

from research_platform_trl.core.checkpoint import StageCheckpoint
from research_platform_trl.core.config import load_yaml_config
from research_platform_trl.core.io_utils import write_json
from research_platform_trl.core.prompting import load_codeact_prompt
from research_platform_trl.data.trajectory_builders import build_teacher_ft_chat_dataset
from research_platform_trl.trainers.sft_common import load_jsonl_dataset
from research_platform_trl.trainers.teacher_sft import train_teacher_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher finetuning on CodeAct trajectories")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config).raw

    out_dir = Path(cfg["output"]["work_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = StageCheckpoint(out_dir / "checkpoints" / "teacher_ft_state.json")

    stage = "build_train_dataset"
    train_jsonl = out_dir / "train_data" / "teacher_ft_train.jsonl"
    if args.resume and ckpt.done(stage):
        kept = ckpt.get_meta("kept_samples", 0)
        print(f"[RESUME] skip {stage}, kept={kept}")
    else:
        prompt = load_codeact_prompt(cfg["prompt"]["file"])
        kept = build_teacher_ft_chat_dataset(
            records_jsonl=Path(cfg["dataset"]["records_jsonl"]).expanduser().resolve(),
            output_jsonl=train_jsonl,
            system_prompt=prompt,
            only_correct=bool(cfg["dataset"].get("only_correct", True)),
        )
        ckpt.set_meta("kept_samples", kept)
        ckpt.mark_done(stage)

    stage = "train_teacher"
    teacher_out = out_dir / "teacher_model"
    if args.resume and ckpt.done(stage):
        print(f"[RESUME] skip {stage}")
    else:
        ds = load_jsonl_dataset(train_jsonl)
        train_cfg = cfg["train"]
        train_teacher_sft(
            model_name_or_path=cfg["teacher"]["model_name_or_path"],
            train_dataset=ds,
            output_dir=teacher_out,
            max_length=int(train_cfg.get("max_length", 4096)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
            epochs=float(train_cfg.get("epochs", 1.0)),
            grad_acc=int(train_cfg.get("grad_acc", 16)),
            per_device_bs=int(train_cfg.get("per_device_batch_size", 1)),
            save_steps=int(train_cfg.get("save_steps", 50)),
            bf16=bool(train_cfg.get("bf16", True)),
            resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"),
        )
        ckpt.mark_done(stage)

    summary = {
        "work_dir": str(out_dir),
        "train_jsonl": str(train_jsonl),
        "teacher_model_dir": str(teacher_out),
        "kept_samples": kept,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({"summary": str(out_dir / 'summary.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
