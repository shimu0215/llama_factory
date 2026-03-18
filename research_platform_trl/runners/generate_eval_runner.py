import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from research_platform_trl.core.agenting import CompressionConfig, create_codeact_agent
from research_platform_trl.core.checkpoint import StageCheckpoint
from research_platform_trl.core.config import load_yaml_config
from research_platform_trl.core.io_utils import append_jsonl, write_json
from research_platform_trl.core.modeling import create_smolagents_model
from research_platform_trl.data.datasets import extract_answer, is_correct, load_qa_examples
from research_platform_trl.tools.registry import create_tools

DEFAULT_QUESTION_LEVEL_INSTRUCTION = (
    "\n\nIMPORTANT: Always provide a 'Thought:' sequence, and a 'Code:\\n```python' sequence ending with "
    "'```<end_code>' sequence, else you will fail. Call the function final_answer(...) exactly once at the end; "
    "never assign to a variable named final_answer. For math problems that are not multiple-choice, always "
    "output the final answer using LaTeX \\boxed{} format. Provide the exact value (e.g., \\boxed{\\frac{9}{14}}), "
    "not a decimal approximation (e.g., \\boxed{0.642857})."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate CodeAct trajectories")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_trajectory(step_dicts: list[dict[str, Any]], final_output: Any) -> tuple[list[dict[str, Any]], str]:
    trajectory: list[dict[str, Any]] = []
    cot_lines: list[str] = []
    for step in step_dicts:
        if "task" in step:
            trajectory.append({"step_type": "user", "content": step["task"]})
            continue
        if "model_output" in step and step.get("model_output"):
            trajectory.append({"step_type": "reasoning", "content": step["model_output"]})
            cot_lines.append(str(step["model_output"]))
        if "code_action" in step and step.get("code_action"):
            trajectory.append({"step_type": "tool_call", "tool": "python", "code": step["code_action"]})
            cot_lines.append(f"```python\n{step['code_action']}\n```")
        if "observations" in step and step.get("observations"):
            trajectory.append({"step_type": "tool_result", "tool": "python", "content": step["observations"]})
            cot_lines.append(f"Observation:\n{step['observations']}")
        if "error" in step and step.get("error"):
            trajectory.append({"step_type": "error", "content": str(step["error"])})
            cot_lines.append(f"Error:\n{step['error']}")
    trajectory.append({"step_type": "final_answer", "content": str(final_output)})
    cot_lines.append(f"Final answer:\n{final_output}")
    return trajectory, "\n\n".join(cot_lines)


def flush_jsonl_buffers(
    records_path: Path,
    contexts_path: Path,
    record_buf: list[dict[str, Any]],
    context_buf: list[dict[str, Any]],
) -> None:
    for rec in record_buf:
        append_jsonl(records_path, rec)
    for ctx in context_buf:
        append_jsonl(contexts_path, ctx)
    record_buf.clear()
    context_buf.clear()


def load_completed_samples(records_path: Path, samples: int) -> dict[int, set[int]]:
    done: dict[int, set[int]] = {}
    if not records_path.exists():
        return done
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            qid = int(obj.get("question_id"))
            sid = int(obj.get("sample_id"))
        except Exception:
            continue
        if sid < 0 or sid >= samples:
            continue
        done.setdefault(qid, set()).add(sid)
    return done


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config).raw

    out_dir = Path(cfg["output"]["work_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = StageCheckpoint(out_dir / "checkpoints" / "generate_eval_state.json")

    stage = "load_data"
    if args.resume and ckpt.done(stage):
        examples = ckpt.get_meta("examples")
    else:
        examples_raw = load_qa_examples(cfg["dataset"])
        examples = [
            {
                "question_id": i,
                "question": ex["question"],
                "answer": ex["answer"],
            }
            for i, ex in enumerate(examples_raw)
        ]
        ckpt.set_meta("examples", examples)
        ckpt.mark_done(stage)

    stage = "run_generation"
    records_path = out_dir / "records.jsonl"
    contexts_path = out_dir / "contexts.jsonl"
    if args.resume and ckpt.done(stage):
        print(f"[RESUME] skip stage={stage}")
    else:
        tools = create_tools(cfg.get("tools", {}))
        model_cfg = cfg["model"].copy()
        model_cfg.setdefault("temperature", cfg["generation"].get("temperature", 0.7))
        model_cfg.setdefault("max_tokens", cfg["generation"].get("max_tokens", 1024))
        samples = int(cfg["generation"].get("num_samples", 1))
        flush_every_questions = int(cfg.get("output", {}).get("flush_every_questions", 5))
        question_level_instruction = str(
            cfg["generation"].get("question_level_instruction", DEFAULT_QUESTION_LEVEL_INSTRUCTION)
        )
        completed_samples = load_completed_samples(records_path, samples) if args.resume else {}
        if args.resume:
            done_q = sum(1 for qid in [ex["question_id"] for ex in examples] if len(completed_samples.get(qid, set())) >= samples)
            print(f"[RESUME] loaded completed samples for {done_q}/{len(examples)} questions")

        total = 0
        first_correct = 0
        processed_questions_since_flush = 0
        record_buf: list[dict[str, Any]] = []
        context_buf: list[dict[str, Any]] = []

        pbar = tqdm(examples, desc="generate_eval", unit="q", dynamic_ncols=True)
        for ex in pbar:
            qid = int(ex["question_id"])
            done_for_q = completed_samples.get(qid, set())
            if len(done_for_q) >= samples:
                continue
            per_q = []
            for sample_id in range(samples):
                if sample_id in done_for_q:
                    continue
                model_client = create_smolagents_model(model_cfg)
                c_cfg_raw = cfg.get("context", {})
                comp_cfg = CompressionConfig(
                    enabled=bool(c_cfg_raw.get("enable_context_compression", True)),
                    mode=str(c_cfg_raw.get("compression_mode", "auto")),
                    model_max_context_tokens=c_cfg_raw.get("model_max_context_tokens"),
                    prompt_budget_tokens=c_cfg_raw.get("prompt_budget_tokens"),
                    recent_steps=int(c_cfg_raw.get("recent_steps", 2)),
                    max_summary_chars=int(c_cfg_raw.get("max_summary_chars", 1200)),
                    max_observation_chars=int(c_cfg_raw.get("max_observation_chars", 1500)),
                    max_step_chars=int(c_cfg_raw.get("max_step_chars", 1200)),
                )
                enable_rolling_memory = bool(c_cfg_raw.get("enable_rolling_memory_code_agent", True))
                code_block_tags = None
                if "```python" in question_level_instruction:
                    code_block_tags = "markdown"
                agent = create_codeact_agent(
                    model_client=model_client,
                    # Use smolagents built-in CodeAgent prompt templates without extra custom instructions.
                    system_prompt="",
                    tools=tools,
                    max_steps=int(cfg["generation"].get("max_steps", 5)),
                    compression_cfg=comp_cfg,
                    enable_rolling_memory=enable_rolling_memory,
                    code_block_tags=code_block_tags,
                )

                error = None
                try:
                    full = agent.run(ex["question"] + question_level_instruction, return_full_result=True)
                    final_answer = str(full.output)
                    steps = full.steps
                    token_usage = full.token_usage.dict() if full.token_usage is not None else None
                    timing = full.timing.dict() if full.timing is not None else {}
                    state = full.state
                except Exception as e:  # noqa: BLE001
                    error = f"{type(e).__name__}: {e}"
                    final_answer = f"[ERROR] {error}"
                    steps = []
                    token_usage = None
                    timing = {}
                    state = {}

                traj, cot = build_trajectory(steps, final_answer)
                pred = extract_answer(final_answer)
                ref = extract_answer(ex["answer"])
                correct = is_correct(pred, ref)
                rec = {
                    "question_id": ex["question_id"],
                    "sample_id": sample_id,
                    "question": ex["question"],
                    "ground_truth": ex["answer"],
                    "final_answer": final_answer,
                    "extracted_prediction": pred,
                    "extracted_ground_truth": ref,
                    "correct": correct,
                    "trajectory": traj,
                    "cot_text": cot,
                    "steps": steps,
                    "state": state,
                    "token_usage": token_usage,
                    "timing": timing,
                    "error": error,
                }
                record_buf.append(rec)
                context_buf.append(
                    {
                        "question_id": ex["question_id"],
                        "sample_id": sample_id,
                        "visible_contexts": getattr(agent, "visible_contexts", []),
                        "correct": correct,
                        "error": error,
                    },
                )
                per_q.append(rec)
                done_for_q.add(sample_id)
                completed_samples[qid] = done_for_q

            total += 1
            processed_questions_since_flush += 1
            if per_q and per_q[0].get("correct", False):
                first_correct += 1
            pbar.set_postfix(first_acc=f"{(first_correct / total) if total else 0.0:.3f}")
            if processed_questions_since_flush >= flush_every_questions:
                flush_jsonl_buffers(records_path, contexts_path, record_buf, context_buf)
                processed_questions_since_flush = 0

        if record_buf or context_buf:
            flush_jsonl_buffers(records_path, contexts_path, record_buf, context_buf)

        ckpt.mark_done(stage)

    stage = "write_summary"
    if args.resume and ckpt.done(stage):
        print(f"[RESUME] skip stage={stage}")
    else:
        first_total = 0
        first_correct = 0
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("sample_id") == 0:
                first_total += 1
                if obj.get("correct", False):
                    first_correct += 1

        summary = {
            "records_path": str(records_path),
            "contexts_path": str(contexts_path),
            "first_sample_accuracy": first_correct / first_total if first_total else 0.0,
            "questions": first_total,
            "config": cfg,
        }
        write_json(out_dir / "summary.json", summary)
        ckpt.mark_done(stage)

    print(json.dumps({"summary": str(out_dir / 'summary.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
