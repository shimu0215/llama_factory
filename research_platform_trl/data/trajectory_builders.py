import json
import random
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_kd_chat_dataset(
    *,
    grouped_records_jsonl: Path,
    output_jsonl: Path,
    system_prompt: str,
    seed: int,
) -> int:
    rng = random.Random(seed)
    kept = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with grouped_records_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            obj = json.loads(line)
            paths = obj.get("paths", [])
            correct = [p for p in paths if p.get("correct", False)]
            if not correct:
                continue
            pick = rng.choice(correct)
            sample = {
                "id": f"q_{obj.get('question_id', kept)}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(obj["question"])},
                    {"role": "assistant", "content": str(pick.get("cot_text", pick.get("final_answer", "")))},
                ],
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def build_teacher_ft_chat_dataset(
    *,
    records_jsonl: Path,
    output_jsonl: Path,
    system_prompt: str,
    only_correct: bool = True,
) -> int:
    rows = _load_jsonl(records_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_jsonl.open("w", encoding="utf-8") as out:
        for row in rows:
            if only_correct and not row.get("correct", False):
                continue
            sample = {
                "id": f"q{row.get('question_id')}_s{row.get('sample_id')}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(row["question"])},
                    {"role": "assistant", "content": str(row.get("cot_text", row.get("final_answer", "")))},
                ],
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1
    return kept
