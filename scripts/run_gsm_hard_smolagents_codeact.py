import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset
from smolagents import CodeAgent, OpenAIModel
from smolagents.monitoring import LogLevel
from tqdm import tqdm


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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--recent-steps", type=int, default=2)
    parser.add_argument("--prompt-budget-tokens", type=int, default=12000)
    parser.add_argument("--max-summary-chars", type=int, default=1200)
    parser.add_argument("--max-observation-chars", type=int, default=1500)
    parser.add_argument("--max-step-chars", type=int, default=1200)
    parser.add_argument("--record-shard-size", type=int, default=0)
    parser.add_argument("--group-shard-size", type=int, default=0)
    parser.add_argument("--context-shard-size", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/qwen3_4b_gsm_hard_smolagents_codeact")
    parser.add_argument("--question-ids-file", default=None, help="JSON file containing selected question ids.")
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


class JsonlShardWriter:
    def __init__(self, output_dir: Path, prefix: str, shard_size: int) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.current_shard = 0
        self.current_count = 0
        self.paths: list[str] = []

    def open_new_shard(self) -> Path:
        if self.shard_size > 0:
            path = self.output_dir / f"{self.prefix}_part_{self.current_shard:03d}.jsonl"
        else:
            path = self.output_dir / f"{self.prefix}.jsonl"
        if str(path) not in self.paths:
            self.paths.append(str(path))
        return path

    def write(self, item: dict[str, Any]) -> None:
        if self.shard_size > 0 and self.current_count >= self.shard_size:
            self.current_shard += 1
            self.current_count = 0
        path = self.open_new_shard()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.current_count += 1


def cleanup_previous_outputs(output_dir: Path) -> None:
    patterns = [
        "gsm_hard_codeact_records*.jsonl",
        "gsm_hard_codeact_grouped*.jsonl",
        "gsm_hard_codeact_contexts*.jsonl",
        "gsm_hard_codeact_grouped*.json",
        "gsm_hard_codeact_summary.json",
    ]
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink()


def truncate_text(text: Any, max_chars: int) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    keep = max(32, max_chars // 2)
    return f"{text[:keep]}\n...[truncated]...\n{text[-keep:]}"


class RollingMemoryCodeAgent(CodeAgent):
    def __init__(
        self,
        *args: Any,
        recent_steps: int,
        prompt_budget_tokens: int,
        max_summary_chars: int,
        max_observation_chars: int,
        max_step_chars: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.recent_steps = recent_steps
        self.prompt_budget_tokens = prompt_budget_tokens
        self.max_summary_chars = max_summary_chars
        self.max_observation_chars = max_observation_chars
        self.max_step_chars = max_step_chars
        self.visible_contexts: list[dict[str, Any]] = []

    def _message_role(self, message: Any) -> Any:
        if isinstance(message, dict):
            return message.get("role")
        return getattr(message, "role", None)

    def _message_content(self, message: Any) -> Any:
        if isinstance(message, dict):
            return message.get("content", [])
        return getattr(message, "content", [])

    def _replace_message_content(self, message: Any, content: Any) -> Any:
        if isinstance(message, dict):
            return {**message, "content": content}
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"content": content})
        if hasattr(message, "copy"):
            return message.copy(update={"content": content})
        message.content = content
        return message

    def _serialize_message(self, message: Any) -> dict[str, Any]:
        content = self._message_content(message)
        if isinstance(content, str):
            serialized_content: Any = content
        else:
            serialized_content = []
            for item in content:
                if isinstance(item, dict):
                    serialized_content.append(item)
                else:
                    serialized_content.append(str(item))
        return {
            "role": str(self._message_role(message)),
            "content": serialized_content,
        }

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        text_chars = 0
        for message in messages:
            content = self._message_content(message)
            if isinstance(content, str):
                text_chars += len(content)
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_chars += len(item.get("text", ""))
        return text_chars // 4

    def _truncate_message_text(self, messages: list[dict[str, Any]], max_chars: int, observation_chars: int) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for message in messages:
            content = self._message_content(message)
            new_content = []
            if isinstance(content, str):
                compacted.append(self._replace_message_content(message, truncate_text(content, max_chars)))
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    new_content.append(item)
                    continue
                text = item.get("text", "")
                limit = observation_chars if "Observation:" in text else max_chars
                new_content.append({**item, "text": truncate_text(text, limit)})
            compacted.append(self._replace_message_content(message, new_content))
        return compacted

    def _summarize_steps(self, steps: list[Any], max_chars: int) -> str:
        if not steps or max_chars <= 0:
            return ""
        lines = ["Previous progress summary:"]
        for idx, step in enumerate(steps, start=1):
            if hasattr(step, "plan") and getattr(step, "plan", None):
                lines.append(f"- Plan {idx}: {truncate_text(step.plan, 160)}")
                continue

            parts = []
            model_output = getattr(step, "model_output", None)
            if model_output:
                parts.append(f"reasoning={truncate_text(model_output, 160)}")
            tool_calls = getattr(step, "tool_calls", None) or []
            if tool_calls:
                parts.append(f"tool_calls={truncate_text(tool_calls[0], 160)}")
            observations = getattr(step, "observations", None)
            if observations:
                parts.append(f"observation={truncate_text(observations, 200)}")
            error = getattr(step, "error", None)
            if error:
                parts.append(f"error={truncate_text(error, 120)}")
            if parts:
                lines.append(f"- Step {idx}: " + "; ".join(parts))

        return truncate_text("\n".join(lines), max_chars)

    def _build_compacted_messages(
        self,
        recent_steps_count: int,
        summary_chars: int,
        observation_chars: int,
        step_chars: int,
    ) -> list[dict[str, Any]]:
        messages = self.memory.system_prompt.to_messages(summary_mode=False)
        steps = list(self.memory.steps)
        if not steps:
            return messages

        first_step = steps[0]
        if hasattr(first_step, "task"):
            messages.extend(first_step.to_messages(summary_mode=False))
            action_steps = steps[1:]
        else:
            action_steps = steps

        recent_steps_count = max(1, min(recent_steps_count, len(action_steps))) if action_steps else 0
        old_steps = action_steps[:-recent_steps_count] if recent_steps_count else action_steps
        recent_steps = action_steps[-recent_steps_count:] if recent_steps_count else []

        summary = self._summarize_steps(old_steps, summary_chars)
        if summary:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": summary}],
                }
            )

        for step in recent_steps:
            messages.extend(step.to_messages(summary_mode=False))

        return self._truncate_message_text(messages, step_chars, observation_chars)

    def write_memory_to_messages(self, summary_mode: Optional[bool] = False) -> list[dict[str, Any]]:
        if summary_mode:
            return super().write_memory_to_messages(summary_mode=True)

        full_messages = super().write_memory_to_messages(summary_mode=False)
        full_estimated_tokens = self._estimate_tokens(full_messages)
        if full_estimated_tokens <= self.prompt_budget_tokens:
            self.visible_contexts.append(
                {
                    "call_index": len(self.visible_contexts),
                    "compressed": False,
                    "compression_mode": "full_memory",
                    "budget_triggered": False,
                    "estimated_tokens_before": full_estimated_tokens,
                    "estimated_tokens_after": full_estimated_tokens,
                    "messages": [self._serialize_message(message) for message in full_messages],
                }
            )
            return full_messages

        recent_steps_count = self.recent_steps
        summary_chars = self.max_summary_chars
        observation_chars = self.max_observation_chars
        step_chars = self.max_step_chars
        messages = self._build_compacted_messages(recent_steps_count, summary_chars, observation_chars, step_chars)

        while self._estimate_tokens(messages) > self.prompt_budget_tokens:
            changed = False
            if summary_chars > 300:
                summary_chars = max(300, summary_chars // 2)
                changed = True
            elif observation_chars > 400:
                observation_chars = max(400, observation_chars // 2)
                changed = True
            elif step_chars > 400:
                step_chars = max(400, step_chars // 2)
                changed = True
            elif recent_steps_count > 1:
                recent_steps_count -= 1
                changed = True
            if not changed:
                break
            messages = self._build_compacted_messages(recent_steps_count, summary_chars, observation_chars, step_chars)

        self.visible_contexts.append(
            {
                "call_index": len(self.visible_contexts),
                "compressed": True,
                "compression_mode": "rolling_summary_window",
                "budget_triggered": True,
                "estimated_tokens_before": full_estimated_tokens,
                "estimated_tokens_after": self._estimate_tokens(messages),
                "recent_steps_used": recent_steps_count,
                "summary_chars_used": summary_chars,
                "observation_chars_used": observation_chars,
                "step_chars_used": step_chars,
                "messages": [self._serialize_message(message) for message in messages],
            }
        )
        return messages


def create_agent(
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_steps: int,
    recent_steps: int,
    prompt_budget_tokens: int,
    max_summary_chars: int,
    max_observation_chars: int,
    max_step_chars: int,
) -> CodeAgent:
    model_client = OpenAIModel(
        model_id=model,
        api_base=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return RollingMemoryCodeAgent(
        tools=[],
        model=model_client,
        instructions=CODEACT_INSTRUCTIONS,
        additional_authorized_imports=["math", "statistics", "fractions", "decimal"],
        max_steps=max_steps,
        recent_steps=recent_steps,
        prompt_budget_tokens=prompt_budget_tokens,
        max_summary_chars=max_summary_chars,
        max_observation_chars=max_observation_chars,
        max_step_chars=max_step_chars,
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
        recent_steps=args.recent_steps,
        prompt_budget_tokens=args.prompt_budget_tokens,
        max_summary_chars=args.max_summary_chars,
        max_observation_chars=args.max_observation_chars,
        max_step_chars=args.max_step_chars,
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
        "visible_contexts": agent.visible_contexts,
    }


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_previous_outputs(output_dir)

    split, raw_examples = load_gsm_hard_examples(args.dataset_name, args.dataset_config, args.split, 0)
    indexed_examples = list(enumerate(raw_examples))
    if args.question_ids_file:
        with Path(args.question_ids_file).open("r", encoding="utf-8") as f:
            selected_ids = set(json.load(f))
        indexed_examples = [(idx, ex) for idx, ex in indexed_examples if idx in selected_ids]

    if args.limit > 0:
        indexed_examples = indexed_examples[: args.limit]

    records_writer = JsonlShardWriter(output_dir, "gsm_hard_codeact_records", args.record_shard_size)
    grouped_writer = JsonlShardWriter(output_dir, "gsm_hard_codeact_grouped", args.group_shard_size)
    context_shard_size = args.context_shard_size if args.context_shard_size > 0 else args.record_shard_size
    contexts_writer = JsonlShardWriter(output_dir, "gsm_hard_codeact_contexts", context_shard_size)

    total_records = 0
    total_questions = 0
    sample_correct = 0
    extracted_count = 0
    question_accuracy_any = 0

    progress = tqdm(indexed_examples, desc="gsm-hard questions", unit="q", dynamic_ncols=True)
    for question_id, example in progress:
        paths = []
        for sample_id in range(args.num_samples):
            record = run_single_sample(
                question_id=question_id,
                sample_id=sample_id,
                question=example["question"],
                answer=example["answer"],
                args=args,
            )
            visible_contexts = record.pop("visible_contexts", [])
            records_writer.write(record)
            contexts_writer.write(
                {
                    "question_id": record["question_id"],
                    "sample_id": record["sample_id"],
                    "question": record["question"],
                    "ground_truth": record["ground_truth"],
                    "correct": record["correct"],
                    "state": record["state"],
                    "visible_contexts": visible_contexts,
                }
            )
            paths.append(record)
            total_records += 1
            if record["correct"]:
                sample_correct += 1
            if record["extracted_prediction"] is not None:
                extracted_count += 1

        grouped_item = {
            "question_id": question_id,
            "question": example["question"],
            "ground_truth": example["answer"],
            "paths": paths,
        }
        grouped_writer.write(grouped_item)
        total_questions += 1
        if any(path["correct"] for path in paths):
            question_accuracy_any += 1
        progress.set_postfix(
            questions=total_questions,
            samples=total_records,
            sample_acc=f"{(sample_correct / total_records) if total_records else 0.0:.3f}",
            any_acc=f"{(question_accuracy_any / total_questions) if total_questions else 0.0:.3f}",
        )
    progress.close()

    metrics = {
        "accuracy": sample_correct / total_records if total_records else 0.0,
        "extraction_rate": extracted_count / total_records if total_records else 0.0,
    }

    summary = {
        "dataset_name": args.dataset_name,
        "split": split,
        "limit": args.limit,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "recent_steps": args.recent_steps,
        "prompt_budget_tokens": args.prompt_budget_tokens,
        "max_summary_chars": args.max_summary_chars,
        "max_observation_chars": args.max_observation_chars,
        "max_step_chars": args.max_step_chars,
        "record_shard_size": args.record_shard_size,
        "group_shard_size": args.group_shard_size,
        "context_shard_size": context_shard_size,
        "model": args.model,
        "base_url": args.base_url,
        "metrics": metrics,
        "sample_accuracy": sample_correct / total_records if total_records else 0.0,
        "question_accuracy_any": question_accuracy_any / total_questions if total_questions else 0.0,
        "records_count": total_records,
        "questions_count": total_questions,
    }

    summary_path = output_dir / "gsm_hard_codeact_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "records_paths": records_writer.paths,
                "grouped_paths": grouped_writer.paths,
                "context_paths": contexts_writer.paths,
                "metrics": metrics,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
