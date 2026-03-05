import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any


THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
ERROR_MARKERS = (
    "AgentParsingError",
    "AgentExecutionError",
    "AgentMaxStepsError",
)

CODEACT_SYSTEM_PROMPT = """Solve the math problem with CodeAct style reasoning.
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
    parser = argparse.ArgumentParser(description="Convert CodeAct records to thought-only SFT messages.")
    parser.add_argument("--input-dir", required=True, help="Directory containing gsm_hard_codeact_records*.jsonl")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--strict", action="store_true", help="Keep only success+correct and drop records with error traces")
    parser.add_argument("--keep-think-tags", action="store_true", help="Keep <think> tags in thought text")
    parser.add_argument("--include-final-as-context", action="store_true", help="Append final answer as a non-trainable context message")
    parser.add_argument("--system-prompt", default=CODEACT_SYSTEM_PROMPT, help="System prompt inserted into each sample")
    return parser.parse_args()


def clean_text(text: Any, keep_think_tags: bool) -> str:
    s = "" if text is None else str(text)
    if not keep_think_tags:
        s = THINK_TAG_RE.sub("", s)
    return s.replace("\r\n", "\n").strip()


def has_error_noise(record: dict[str, Any]) -> bool:
    cot = str(record.get("cot_text", ""))
    if any(marker in cot for marker in ERROR_MARKERS):
        return True
    for step in record.get("trajectory", []):
        if step.get("step_type") == "error":
            return True
    return False


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(str(input_dir / "gsm_hard_codeact_records*.jsonl")))
    out: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def build_messages(record: dict[str, Any], keep_think_tags: bool, include_final_as_context: bool, system_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": clean_text(record.get("question", ""), keep_think_tags=True)},
    ]
    assistant_chunks: list[str] = []
    pending_code = ""

    def flush_assistant():
        if assistant_chunks:
            text = "\n\n".join([c for c in assistant_chunks if c]).strip()
            if text:
                messages.append({"role": "assistant", "content": text})
            assistant_chunks.clear()

    for step in record.get("trajectory", []):
        st = step.get("step_type")
        if st == "reasoning":
            thought = clean_text(step.get("content", ""), keep_think_tags=keep_think_tags)
            if thought:
                assistant_chunks.append(f"Thought:\n{thought}")
        elif st == "tool_call":
            code = clean_text(step.get("code", ""), keep_think_tags=True)
            if code:
                pending_code = code
        elif st == "tool_result":
            obs = clean_text(step.get("content", ""), keep_think_tags=True)
            flush_assistant()
            obs_parts: list[str] = []
            if pending_code:
                obs_parts.append(f"Code:\n```python\n{pending_code}\n```")
            if obs:
                obs_parts.append(f"Execution/Observation:\n{obs}")
            if obs_parts:
                # Observation turn is context-only and keeps user/assistant alternation valid.
                messages.append({"role": "observation", "content": "\n\n".join(obs_parts)})
            pending_code = ""
        elif st == "final_answer":
            fa = clean_text(step.get("content", ""), keep_think_tags=keep_think_tags)
            if fa:
                assistant_chunks.append(fa)
            elif include_final_as_context:
                fa_ctx = clean_text(step.get("content", ""), keep_think_tags=True)
                if fa_ctx:
                    assistant_chunks.append(f"Reference final answer:\n{fa_ctx}")

    flush_assistant()

    # Remove trailing observation so the final supervised turn is assistant.
    while len(messages) > 2 and messages[-1]["role"] == "observation":
        messages.pop()

    # Ensure odd number of prompt turns + 1 response turn for SFT processor.
    if len(messages) >= 3 and messages[-1]["role"] != "assistant":
        messages.append({"role": "assistant", "content": "Thought:\n(omitted)"})

    return messages


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_dir)
    total = len(records)
    kept = 0
    dropped_not_correct = 0
    dropped_not_success = 0
    dropped_error_noise = 0
    dropped_no_thought = 0

    with output_path.open("w", encoding="utf-8") as out:
        for r in records:
            if not r.get("correct", False):
                dropped_not_correct += 1
                continue
            if r.get("state") != "success":
                dropped_not_success += 1
                continue
            if args.strict and has_error_noise(r):
                dropped_error_noise += 1
                continue

            messages = build_messages(
                r,
                keep_think_tags=args.keep_think_tags,
                include_final_as_context=args.include_final_as_context,
                system_prompt=args.system_prompt,
            )
            trainable_turns = sum(1 for m in messages if m["role"] == "assistant")
            if trainable_turns == 0:
                dropped_no_thought += 1
                continue

            sample = {
                "id": f"gsmhard_q{r.get('question_id')}_s{r.get('sample_id')}",
                "question_id": r.get("question_id"),
                "sample_id": r.get("sample_id"),
                "messages": messages,
                "meta": {
                    "ground_truth": r.get("ground_truth"),
                    "extracted_ground_truth": r.get("extracted_ground_truth"),
                    "extracted_prediction": r.get("extracted_prediction"),
                    "state": r.get("state"),
                    "correct": r.get("correct"),
                    "token_usage": r.get("token_usage"),
                    "timing": r.get("timing"),
                    "trainable_turns": trainable_turns,
                },
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1

    stats = {
        "input_dir": str(input_dir),
        "output": str(output_path),
        "strict": args.strict,
        "keep_think_tags": args.keep_think_tags,
        "include_final_as_context": args.include_final_as_context,
        "total_records": total,
        "kept_records": kept,
        "dropped_not_correct": dropped_not_correct,
        "dropped_not_success": dropped_not_success,
        "dropped_error_noise": dropped_error_noise,
        "dropped_no_thought": dropped_no_thought,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
