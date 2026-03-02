import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset
from smolagents import CodeAgent, OpenAIModel
from smolagents.monitoring import LogLevel


ROOT = Path(__file__).resolve().parents[1]

CODEACT_INSTRUCTIONS = """
Solve the math problem with CodeAct style reasoning.
You only have Python execution available through your code blocks. Do not use web search or external tools.
For intermediate reasoning:
- Think briefly but concretely.
- Write Python code when computation helps.
- Use print(...) for intermediate values you want to inspect.
- Keep the code correct and minimal.
For the final step:
- Use final_answer(...) exactly once.
- The final answer must be a bare number or a short numeric string, with no extra explanation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GSM-hard CodeAct trajectories with smolagents.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="0")
    parser.add_argument("--model", default="test")
    parser.add_argument("--dataset-name", default="reasoning-machines/gsm-hard")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default="outputs/qwen3_4b_gsm_hard_smolagents_codeact")
    return parser.parse_args()


def extract_answer(text: Any) -> Optional[str]:
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return None

    boxed = re.findall(r"\\boxed{([^{}]*)}", text)
    if boxed:
        return boxed[-1].strip()

    numbers = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        return numbers[-1]

    return None


def compare_answers(predicted: Optional[str], reference: Optional[str]) -> bool:
    if predicted is None or reference is None:
        return False
    if predicted.strip() == reference.strip():
        return True
    try:
        return abs(float(predicted) - float(reference)) < 1e-6
    except ValueError:
        return predicted.strip() == reference.strip()


def evaluate_predictions(predictions: list[str], references: list[str]) -> dict[str, float]:
    correct = 0
    extracted_count = 0
    for pred, ref in zip(predictions, references):
        extracted_pred = extract_answer(pred)
        extracted_ref = extract_answer(ref)
        if extracted_pred is not None:
            extracted_count += 1
        if compare_answers(extracted_pred, extracted_ref):
            correct += 1
    total = len(references)
    return {
        "accuracy": correct / total if total else 0.0,
        "extraction_rate": extracted_count / total if total else 0.0,
    }


def load_gsm_hard_examples(dataset_name: str, dataset_config: str, split: Optional[str], limit: int) -> tuple[str, list[dict[str, Any]]]:
    dataset_dict = load_dataset(dataset_name, dataset_config)
    if split is None:
        for candidate in ("test", "validation", "train"):
            if candidate in dataset_dict:
                split = candidate
                break
    if split is None:
        raise ValueError(f"Cannot infer split from dataset keys: {list(dataset_dict.keys())}")

    dataset = dataset_dict[split]
    if "input" in dataset.column_names and "question" not in dataset.column_names:
        dataset = dataset.rename_column("input", "question")
    if "target" in dataset.column_names and "answer" not in dataset.column_names:
        dataset = dataset.rename_column("target", "answer")

    if limit <= 0:
        limit = len(dataset)

    return split, [dataset[idx] for idx in range(min(limit, len(dataset)))]


def create_agent(base_url: str, api_key: str, model: str, temperature: float, max_tokens: int, max_steps: int) -> CodeAgent:
    model_client = OpenAIModel(
        model_id=model,
        api_base=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return CodeAgent(
        tools=[],
        model=model_client,
        instructions=CODEACT_INSTRUCTIONS,
        additional_authorized_imports=["math", "statistics", "fractions", "decimal"],
        max_steps=max_steps,
        verbosity_level=LogLevel.ERROR,
        stream_outputs=False,
        use_structured_outputs_internally=False,
        code_block_tags="markdown",
        return_full_result=True,
    )


def build_structured_trajectory(step_dicts: list[dict[str, Any]], final_output: Any) -> tuple[list[dict[str, Any]], str]:
    trajectory: list[dict[str, Any]] = []
    cot_lines: list[str] = []

    for step in step_dicts:
        if "task" in step:
            trajectory.append({"step_type": "user", "content": step["task"]})
            continue

        if "plan" in step:
            trajectory.append({"step_type": "planning", "content": step["plan"]})
            cot_lines.append(f"Planning:\n{step['plan']}")
            continue

        model_output = step.get("model_output")
        if model_output:
            trajectory.append({"step_type": "reasoning", "content": model_output})
            cot_lines.append(model_output)

        code_action = step.get("code_action")
        if code_action:
            trajectory.append({"step_type": "tool_call", "tool": "python", "code": code_action})
            cot_lines.append(f"```python\n{code_action}\n```")

        observations = step.get("observations")
        if observations:
            trajectory.append({"step_type": "tool_result", "tool": "python", "content": observations})
            cot_lines.append(f"Observation:\n{observations}")

        error = step.get("error")
        if error:
            trajectory.append({"step_type": "error", "content": error})
            cot_lines.append(f"Error:\n{error}")

    trajectory.append({"step_type": "final_answer", "content": str(final_output)})
    cot_lines.append(f"Final answer:\n{final_output}")
    return trajectory, "\n\n".join(cot_lines)


def run_single_sample(
    question_id: int,
    sample_id: int,
    question: str,
    answer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    agent = create_agent(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
    )
    full_result = agent.run(question, return_full_result=True)
    final_answer = str(full_result.output)
    trajectory, cot_text = build_structured_trajectory(full_result.steps, full_result.output)

    extracted_prediction = extract_answer(final_answer)
    extracted_ground_truth = extract_answer(answer)
    correct = compare_answers(extracted_prediction, extracted_ground_truth)

    return {
        "question_id": question_id,
        "sample_id": sample_id,
        "question": question,
        "ground_truth": answer,
        "final_answer": final_answer,
        "extracted_prediction": extracted_prediction,
        "extracted_ground_truth": extracted_ground_truth,
        "correct": correct,
        "state": full_result.state,
        "token_usage": full_result.token_usage.dict() if full_result.token_usage is not None else None,
        "timing": full_result.timing.dict(),
        "steps": full_result.steps,
        "trajectory": trajectory,
        "cot_text": cot_text,
    }


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split, examples = load_gsm_hard_examples(args.dataset_name, args.dataset_config, args.split, args.limit)

    records: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []

    for question_id, example in enumerate(examples):
        paths = []
        for sample_id in range(args.num_samples):
            record = run_single_sample(
                question_id=question_id,
                sample_id=sample_id,
                question=example["question"],
                answer=example["answer"],
                args=args,
            )
            records.append(record)
            paths.append(record)

        grouped.append(
            {
                "question_id": question_id,
                "question": example["question"],
                "ground_truth": example["answer"],
                "paths": paths,
            }
        )

    predictions = [item["final_answer"] for item in records]
    references = [item["ground_truth"] for item in records]
    metrics = evaluate_predictions(predictions, references)

    summary = {
        "dataset_name": args.dataset_name,
        "split": split,
        "limit": args.limit,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "model": args.model,
        "base_url": args.base_url,
        "metrics": metrics,
        "sample_accuracy": sum(1 for item in records if item["correct"]) / len(records),
        "question_accuracy_any": sum(1 for group in grouped if any(path["correct"] for path in group["paths"])) / len(grouped),
        "records_count": len(records),
    }

    summary_path = output_dir / "gsm_hard_codeact_summary.json"
    records_path = output_dir / "gsm_hard_codeact_records.jsonl"
    grouped_path = output_dir / "gsm_hard_codeact_grouped.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    grouped_path.write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "records_path": str(records_path),
                "grouped_path": str(grouped_path),
                "metrics": metrics,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
