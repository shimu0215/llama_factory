import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute short Python code for arithmetic or symbolic calculation and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Short Python code. Use print(...) to emit the result you want to inspect.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = (
    "You are solving GSM-hard math problems. "
    "You must call the python_exec tool before giving the final answer. "
    "Use the tool for arithmetic, equation checking, or sanity checks. "
    "After receiving the tool result, explain briefly and end with `Final answer: <number>`."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSM-hard tool-use inference with python_exec.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="0")
    parser.add_argument("--model", default="test")
    parser.add_argument("--dataset-name", default="reasoning-machines/gsm-hard")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--save-cot-count", type=int, default=10)
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/gsm_hard_python_exec_eval")
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


def load_gsm_hard_examples(
    dataset_name: str, dataset_config: str, split: Optional[str], limit: int
) -> Tuple[str, list[dict[str, Any]]]:
    from datasets import load_dataset

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

    return split, [dataset[idx] for idx in range(min(limit, len(dataset)))]


def execute_python(code: str, timeout: int) -> dict[str, Any]:
    wrapped_code = build_wrapped_code(code)
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", wrapped_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "returncode": completed.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Execution timed out after {timeout} seconds.", "returncode": -1}


def build_wrapped_code(code: str) -> str:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return code

    if not tree.body:
        return code

    last_stmt = tree.body[-1]
    if isinstance(last_stmt, ast.Expr):
        tree.body[-1] = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[last_stmt.value],
                keywords=[],
            )
        )
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    return code


def parse_tool_arguments(raw_arguments: Any) -> dict[str, str]:
    arguments = raw_arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"code": arguments}

    if isinstance(arguments, str):
        return {"code": arguments}

    if isinstance(arguments, dict):
        if "code" in arguments and isinstance(arguments["code"], str):
            return {"code": arguments["code"]}
        return {"code": json.dumps(arguments, ensure_ascii=False)}

    return {"code": str(arguments)}


def message_to_dict(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", None)
    reasoning_content = getattr(message, "reasoning_content", None)
    tool_calls = []

    if getattr(message, "tool_calls", None):
        for tool_call in message.tool_calls:
            tool_calls.append(
                {
                    "id": getattr(tool_call, "id", None),
                    "name": tool_call.function.name,
                    "arguments": parse_tool_arguments(tool_call.function.arguments),
                }
            )

    return {
        "role": getattr(message, "role", "assistant"),
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
    }


def run_single_example(
    client: Any,
    model: str,
    question: str,
    answer: str,
    max_tool_rounds: int,
    timeout: int,
) -> dict[str, Any]:
    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trajectory: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    tool_rounds = 0
    final_response = ""
    final_reasoning = None

    while True:
        result = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.0,
        )
        assistant_message = result.choices[0].message
        trajectory.append(message_to_dict(assistant_message))
        messages.append(assistant_message)

        final_response = assistant_message.content or ""
        final_reasoning = getattr(assistant_message, "reasoning_content", None)

        if not assistant_message.tool_calls:
            break

        tool_rounds += 1
        if tool_rounds > max_tool_rounds:
            break

        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            arguments = parse_tool_arguments(tool_call.function.arguments)
            if name != "python_exec":
                tool_result = {"stdout": "", "stderr": f"Unsupported tool: {name}", "returncode": -1}
            else:
                tool_result = execute_python(arguments["code"], timeout=timeout)

            messages.append({"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False)})
            trajectory.append(
                {
                    "role": "tool",
                    "name": name,
                    "arguments": arguments,
                    "content": tool_result,
                }
            )

    extracted_prediction = extract_answer(final_response)
    extracted_reference = extract_answer(answer)

    return {
        "question": question,
        "ground_truth": answer,
        "prediction": final_response,
        "reasoning_content": final_reasoning,
        "extracted_prediction": extracted_prediction,
        "extracted_ground_truth": extracted_reference,
        "correct": compare_answers(extracted_prediction, extracted_reference),
        "tool_rounds": tool_rounds,
        "trajectory": trajectory,
    }


def main() -> None:
    from openai import OpenAI

    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    split, examples = load_gsm_hard_examples(args.dataset_name, args.dataset_config, args.split, args.limit)

    results = []
    for example in examples:
        results.append(
            run_single_example(
                client=client,
                model=args.model,
                question=example["question"],
                answer=example["answer"],
                max_tool_rounds=args.max_tool_rounds,
                timeout=args.timeout,
            )
        )

    predictions = [item["prediction"] for item in results]
    references = [item["ground_truth"] for item in results]
    metrics = evaluate_predictions(predictions, references)

    summary = {
        "dataset_name": args.dataset_name,
        "split": split,
        "limit": args.limit,
        "save_cot_count": args.save_cot_count,
        "model": args.model,
        "base_url": args.base_url,
        "metrics": metrics,
        "tool_call_rate": sum(1 for item in results if item["tool_rounds"] > 0) / len(results),
        "samples": [{key: value for key, value in item.items() if key != "trajectory"} for item in results],
    }

    result_path = output_dir / "gsm_hard_python_exec_results.json"
    cot_path = output_dir / "gsm_hard_python_exec_cot.jsonl"

    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with cot_path.open("w", encoding="utf-8") as f:
        for item in results[: args.save_cot_count]:
            f.write(
                json.dumps(
                    {
                        "question": item["question"],
                        "ground_truth": item["ground_truth"],
                        "prediction": item["prediction"],
                        "reasoning_content": item["reasoning_content"],
                        "trajectory": item["trajectory"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(json.dumps({"result_path": str(result_path), "cot_path": str(cot_path), "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
