import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


def extract_answer(text: Any) -> str | None:
    if text is None:
        return None
    s = str(text)
    boxed = re.findall(r"\\boxed{([^{}]*)}", s)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"-?\d*\.?\d+", s.replace(",", ""))
    if nums:
        return nums[-1]
    return None


def is_correct(pred: str | None, ref: str | None) -> bool:
    if pred is None or ref is None:
        return False
    if pred.strip() == ref.strip():
        return True
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except Exception:
        return pred.strip() == ref.strip()


def load_qa_examples(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if "path" in cfg:
        ds = load_dataset(cfg["path"], cfg.get("config", "default"))
    else:
        raise ValueError("dataset.path is required")

    split = cfg.get("split")
    if split is None:
        for candidate in ("test", "validation", "train"):
            if candidate in ds:
                split = candidate
                break
    if split is None:
        raise RuntimeError(f"Cannot infer split from dataset keys: {list(ds.keys())}")

    dsplit = ds[split]
    if "input" in dsplit.column_names and "question" not in dsplit.column_names:
        dsplit = dsplit.rename_column("input", "question")
    if "target" in dsplit.column_names and "answer" not in dsplit.column_names:
        dsplit = dsplit.rename_column("target", "answer")

    examples = [dsplit[i] for i in range(len(dsplit))]

    ids_file = cfg.get("question_ids_file")
    if ids_file:
        ids = set(__import__("json").loads(Path(ids_file).read_text(encoding="utf-8")))
        examples = [ex for i, ex in enumerate(examples) if i in ids]

    limit = int(cfg.get("limit", 0))
    if limit > 0:
        examples = examples[:limit]

    return examples
