from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


@dataclass
class SFTRunResult:
    output_dir: Path
    train_size: int


def load_jsonl_dataset(path: str | Path) -> Dataset:
    p = Path(path).expanduser().resolve()
    ds = load_dataset("json", data_files=str(p))["train"]
    return ds


def train_sft_chat(
    *,
    model_name_or_path: str,
    train_dataset: Dataset,
    output_dir: Path,
    max_length: int,
    learning_rate: float,
    epochs: float,
    grad_acc: int,
    per_device_bs: int,
    save_steps: int,
    bf16: bool,
    deepspeed: str | dict[str, Any] | None = None,
    resume_from_checkpoint: str | None = None,
) -> SFTRunResult:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    args = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        gradient_accumulation_steps=grad_acc,
        per_device_train_batch_size=per_device_bs,
        save_steps=save_steps,
        save_strategy="steps",
        logging_steps=10,
        max_length=max_length,
        bf16=bf16,
        deepspeed=deepspeed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model_name_or_path,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    return SFTRunResult(output_dir=output_dir, train_size=len(train_dataset))
