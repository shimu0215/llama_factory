import argparse
import importlib.util
import inspect
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from datasets import load_dataset
from smolagents.agents import MultiStepAgent
from smolagents import CodeAgent, OpenAIModel
from smolagents.monitoring import LogLevel
from tqdm import tqdm

try:
    from smolagents import WikipediaRetrieverTool, DuckDuckGoSearchTool
except Exception:  # noqa: BLE001
    WikipediaRetrieverTool = None
    DuckDuckGoSearchTool = None

try:
    import sympy as sp
except Exception:  # noqa: BLE001
    sp = None

ROOT = Path(__file__).resolve().parents[1]

# Aligned to AgentDistill exps_research/unified_framework/processors/agent.py instruction block.
AGENTDISTILL_STYLE_INSTRUCTION = (
    "\n\nIMPORTANT: Always provide a 'Thought:' sequence, and a 'Code:\n```python' "
    "sequence ending with '```<end_code>' sequence, else you will fail. "
    "Call the function final_answer(...) exactly once at the end; never assign to a variable named final_answer. "
    "For math problems that are not multiple-choice, always output the final answer "
    "using LaTeX \\boxed{} format. Provide the exact value (e.g., \\boxed{\\frac{9}{14}}), "
    "not a decimal approximation (e.g., \\boxed{0.642857})."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSM-hard eval aligned with AgentDistill-style settings.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="0")
    parser.add_argument("--model", default="test")

    parser.add_argument("--dataset-name", default="reasoning-machines/gsm-hard")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-path",
        default="",
        help="Optional local JSON dataset path (expects AgentDistill-style {'examples': [...]} or a plain list).",
    )
    parser.add_argument("--limit", type=int, default=0)

    # AgentDistill-like defaults
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)

    parser.add_argument(
        "--tool-mode",
        choices=["python_only", "wikipedia", "duckduckgo"],
        default="python_only",
        help="Tool setup for alignment experiments.",
    )
    parser.add_argument(
        "--strict-tool-mode",
        action="store_true",
        help="Fail fast if requested tool mode is unavailable instead of falling back.",
    )

    parser.add_argument("--output-dir", default="outputs/gsm_hard_agentdistill_aligned_eval")
    parser.add_argument("--record-shard-size", type=int, default=1000)
    return parser.parse_args()


def _load_agentdistill_math_helpers() -> tuple[Optional[Callable[..., Any]], Optional[Callable[..., Any]]]:
    candidates = [
        Path("/scratch/wzhao20/AgentDistill/exps_research/unified_framework/processors/qwen_math_parser.py"),
        Path("/Users/shimu/Downloads/DOGe-main/AgentDistill/exps_research/unified_framework/processors/qwen_math_parser.py"),
    ]
    for parser_path in candidates:
        if not parser_path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("agentdistill_qwen_math_parser", parser_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[assignment]
            ad_extract = getattr(module, "extract_answer", None)
            ad_math_equal = getattr(module, "math_equal", None)
            if callable(ad_extract) and callable(ad_math_equal):
                return ad_extract, ad_math_equal
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Failed to load AgentDistill parser from {parser_path}: {exc}")
    return None, None


AD_EXTRACT_ANSWER, AD_MATH_EQUAL = _load_agentdistill_math_helpers()


def extract_answer(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text)

    if AD_EXTRACT_ANSWER is not None:
        try:
            extracted = AD_EXTRACT_ANSWER(s)  # type: ignore[misc]
            if extracted is None:
                return None
            extracted_s = str(extracted).strip()
            return extracted_s or None
        except Exception:  # noqa: BLE001
            pass

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


def _try_sympy_equal(a: str, b: str) -> bool:
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

    if AD_MATH_EQUAL is not None:
        try:
            return bool(AD_MATH_EQUAL(predicted, reference, timeout=True))  # type: ignore[misc]
        except Exception:  # noqa: BLE001
            pass

    p = predicted.strip()
    r = reference.strip()
    if p == r:
        return True

    try:
        return abs(float(p) - float(r)) < 1e-6
    except Exception:  # noqa: BLE001
        pass

    return _try_sympy_equal(p, r)


def make_model(base_url: str, api_key: str, model: str, temperature: float, max_tokens: int) -> OpenAIModel:
    return OpenAIModel(
        model_id=model,
        api_base=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def make_tools(tool_mode: str, strict: bool) -> list[Any]:
    if tool_mode == "python_only":
        return []
    if tool_mode == "wikipedia":
        if WikipediaRetrieverTool is None:
            msg = "WikipediaRetrieverTool is unavailable in current smolagents install."
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg + " Fallback to python_only.")
            return []
        return [WikipediaRetrieverTool()]
    if tool_mode == "duckduckgo":
        if DuckDuckGoSearchTool is None:
            msg = "DuckDuckGoSearchTool is unavailable in current smolagents install."
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg + " Fallback to python_only.")
            return []
        return [DuckDuckGoSearchTool()]
    raise ValueError(f"unknown tool_mode: {tool_mode}")


def make_code_agent(tools: list[Any], model_client: OpenAIModel, max_steps: int) -> CodeAgent:
    kwargs: dict[str, Any] = {
        "tools": tools,
        "model": model_client,
        "max_steps": max_steps,
        "additional_authorized_imports": ["numpy", "sympy", "numpy.linalg"],
        "verbosity_level": LogLevel.ERROR,
        "set_timeout": True,
    }
    # Keep kwargs accepted by CodeAgent and by its MultiStepAgent base class.
    allowed = set(inspect.signature(CodeAgent.__init__).parameters) | set(
        inspect.signature(MultiStepAgent.__init__).parameters
    )
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return CodeAgent(**filtered)


def load_examples(dataset_name: str, dataset_config: str, split: str, limit: int) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, dataset_config)
    if split not in ds:
        fallback = None
        for candidate in ("test", "validation", "train"):
            if candidate in ds:
                fallback = candidate
                break
        if fallback is None:
            raise ValueError(f"split {split} not in dataset keys {list(ds.keys())}")
        print(f"[WARN] split={split} not found, fallback to split={fallback}")
        split = fallback
    dsplit = ds[split]

    if "input" in dsplit.column_names and "question" not in dsplit.column_names:
        dsplit = dsplit.rename_column("input", "question")
    if "target" in dsplit.column_names and "answer" not in dsplit.column_names:
        dsplit = dsplit.rename_column("target", "answer")

    rows = [dsplit[i] for i in range(len(dsplit))]
    if limit > 0:
        rows = rows[:limit]
    return rows


def load_examples_from_json(data_path: Path, limit: int) -> list[dict[str, Any]]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("examples"), list):
            rows = data["examples"]
        else:
            raise ValueError(f"Unsupported JSON object format in {data_path}")
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported dataset JSON type in {data_path}: {type(data).__name__}")

    if limit > 0:
        rows = rows[:limit]
    return rows


class JsonlShardWriter:
    def __init__(self, out_dir: Path, prefix: str, shard_size: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.shard_idx = 0
        self.count = 0
        self.paths: list[str] = []

    def _path(self) -> Path:
        if self.shard_size > 0:
            p = self.out_dir / f"{self.prefix}_part_{self.shard_idx:03d}.jsonl"
        else:
            p = self.out_dir / f"{self.prefix}.jsonl"
        if str(p) not in self.paths:
            self.paths.append(str(p))
        return p

    def write(self, obj: dict[str, Any]) -> None:
        if self.shard_size > 0 and self.count >= self.shard_size:
            self.shard_idx += 1
            self.count = 0
        p = self._path()
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.count += 1


def majority_vote_answer(preds: list[Optional[str]]) -> Optional[str]:
    cleaned = [p for p in preds if p is not None]
    if not cleaned:
        return None
    cnt = Counter(cleaned)
    return cnt.most_common(1)[0][0]


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_path:
        data_path = Path(args.data_path).expanduser()
        if not data_path.is_absolute():
            data_path = (ROOT / data_path).resolve()
        examples = load_examples_from_json(data_path, args.limit)
        dataset_name = str(data_path)
        dataset_config = "local_json"
        split = "file"
    else:
        examples = load_examples(args.dataset_name, args.dataset_config, args.split, args.limit)
        dataset_name = args.dataset_name
        dataset_config = args.dataset_config
        split = args.split
    tools = make_tools(args.tool_mode, args.strict_tool_mode)

    rec_writer = JsonlShardWriter(out_dir, "gsm_hard_agentdistill_aligned_records", args.record_shard_size)
    grp_writer = JsonlShardWriter(out_dir, "gsm_hard_agentdistill_aligned_grouped", max(1, args.record_shard_size // 2))

    total_q = 0
    first_correct = 0
    any_correct = 0
    maj_correct = 0

    pbar = tqdm(examples, desc="gsm-hard aligned eval", unit="q", dynamic_ncols=True)
    for qid, ex in enumerate(pbar):
        paths = []
        per_pred: list[Optional[str]] = []
        ref = extract_answer(ex["answer"])

        for sid in range(args.num_samples):
            model_client = make_model(args.base_url, args.api_key, args.model, args.temperature, args.max_tokens)
            agent = make_code_agent(tools=tools, model_client=model_client, max_steps=args.max_steps)

            query = str(ex["question"]) + AGENTDISTILL_STYLE_INSTRUCTION
            err = None
            try:
                full = agent.run(query, return_full_result=True)
                final_answer = str(full.output)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                final_answer = f"[ERROR] {err}"
                full = None

            pred = extract_answer(final_answer)
            per_pred.append(pred)
            ok = compare_answers(pred, ref)

            rec = {
                "question_id": qid,
                "sample_id": sid,
                "question": ex["question"],
                "ground_truth": ex["answer"],
                "extracted_ground_truth": ref,
                "final_answer": final_answer,
                "extracted_prediction": pred,
                "correct": ok,
                "error": err,
                "steps": full.steps if full is not None else [],
                "token_usage": full.token_usage.dict() if (full is not None and full.token_usage is not None) else None,
                "timing": full.timing.dict() if full is not None else {},
                "state": full.state if full is not None else {},
            }
            rec_writer.write(rec)
            paths.append(rec)

        total_q += 1
        if paths and paths[0]["correct"]:
            first_correct += 1
        if any(p["correct"] for p in paths):
            any_correct += 1
        maj_pred = majority_vote_answer(per_pred)
        if compare_answers(maj_pred, ref):
            maj_correct += 1

        grp_writer.write(
            {
                "question_id": qid,
                "question": ex["question"],
                "ground_truth": ex["answer"],
                "paths": paths,
                "majority_prediction": maj_pred,
            }
        )

        pbar.set_postfix(
            first_acc=f"{(first_correct / total_q) if total_q else 0.0:.3f}",
            any_acc=f"{(any_correct / total_q) if total_q else 0.0:.3f}",
            maj_acc=f"{(maj_correct / total_q) if total_q else 0.0:.3f}",
        )

    summary = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "data_path": args.data_path or None,
        "limit": args.limit,
        "model": args.model,
        "base_url": args.base_url,
        "tool_mode": args.tool_mode,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "metrics": {
            "first_sample_accuracy": first_correct / total_q if total_q else 0.0,
            "any_sample_accuracy": any_correct / total_q if total_q else 0.0,
            "majority_vote_accuracy": maj_correct / total_q if total_q else 0.0,
        },
        "records_paths": rec_writer.paths,
        "grouped_paths": grp_writer.paths,
    }
    summary_path = out_dir / "gsm_hard_agentdistill_aligned_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
