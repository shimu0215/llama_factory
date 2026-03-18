import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

try:
    import sympy as sp
except Exception:  # noqa: BLE001
    sp = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute TRL GSM-hard accuracy with AgentDistill-aligned answer extraction and comparison."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="TRL generated data directory containing records.jsonl and summary.json.",
    )
    parser.add_argument(
        "--output-name",
        default="summary_agentdistill_rejudge.json",
        help="Output JSON filename written under input-dir.",
    )
    return parser.parse_args()


def extract_answer(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text)

    answer_tag = re.findall(r"<answer>\s*(.*?)\s*</answer>", s, flags=re.DOTALL)
    if answer_tag:
        s = answer_tag[-1]

    boxed = re.findall(r"\\boxed{([^{}]*)}", s)
    if boxed:
        return boxed[-1].strip()

    numbers = re.findall(r"-?\d*\.?\d+", s.replace(",", ""))
    if numbers:
        return numbers[-1]

    return None


def try_sympy_equal(a: str, b: str) -> bool:
    if sp is None:
        return False
    try:
        a_expr = sp.sympify(a)
        b_expr = sp.sympify(b)
        return sp.simplify(a_expr - b_expr) == 0
    except Exception:  # noqa: BLE001
        return False


def compare_answers(predicted: Optional[str], reference: Optional[str]) -> bool:
    if predicted is None or reference is None:
        return False

    p = predicted.strip()
    r = reference.strip()
    if p == r:
        return True

    try:
        return abs(float(p) - float(r)) < 1e-6
    except Exception:  # noqa: BLE001
        pass

    return try_sympy_equal(p, r)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def majority_vote_answer(preds: list[Optional[str]]) -> Optional[str]:
    cleaned = [p for p in preds if p is not None]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    records_path = input_dir / "records.jsonl"
    original_summary_path = input_dir / "summary.json"
    output_path = input_dir / args.output_name

    if not records_path.exists():
        raise FileNotFoundError(f"Missing records file: {records_path}")

    records = load_jsonl(records_path)
    by_question: dict[int, list[dict[str, Any]]] = defaultdict(list)

    sample_correct = 0
    extracted_count = 0
    for rec in records:
        final_answer = rec.get("final_answer")
        ground_truth = rec.get("ground_truth")
        pred = extract_answer(final_answer)
        ref = extract_answer(ground_truth)
        correct = compare_answers(pred, ref)

        rec["agentdistill_extracted_prediction"] = pred
        rec["agentdistill_extracted_ground_truth"] = ref
        rec["agentdistill_correct"] = correct

        by_question[int(rec["question_id"])].append(rec)
        if correct:
            sample_correct += 1
        if pred is not None:
            extracted_count += 1

    first_correct = 0
    any_correct = 0
    maj_correct = 0
    questions = 0
    for question_id in sorted(by_question):
        paths = sorted(by_question[question_id], key=lambda item: int(item.get("sample_id", 0)))
        questions += 1
        if paths and paths[0]["agentdistill_correct"]:
            first_correct += 1
        if any(path["agentdistill_correct"] for path in paths):
            any_correct += 1

        ref = paths[0]["agentdistill_extracted_ground_truth"] if paths else None
        maj_pred = majority_vote_answer([path["agentdistill_extracted_prediction"] for path in paths])
        if compare_answers(maj_pred, ref):
            maj_correct += 1

    original_summary = None
    if original_summary_path.exists():
        original_summary = json.loads(original_summary_path.read_text(encoding="utf-8"))

    summary = {
        "input_dir": str(input_dir),
        "records_path": str(records_path),
        "judging_mode": "agentdistill_aligned",
        "sympy_available": sp is not None,
        "samples": len(records),
        "questions": questions,
        "metrics": {
            "sample_accuracy": sample_correct / len(records) if records else 0.0,
            "extraction_rate": extracted_count / len(records) if records else 0.0,
            "first_sample_accuracy": first_correct / questions if questions else 0.0,
            "any_sample_accuracy": any_correct / questions if questions else 0.0,
            "majority_vote_accuracy": maj_correct / questions if questions else 0.0,
        },
        "original_summary": original_summary,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
