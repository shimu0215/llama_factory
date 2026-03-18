import importlib.util
import json
import re
import warnings
from pathlib import Path
from typing import Any

from datasets import load_dataset
try:
    import sympy as sp
except Exception:  # noqa: BLE001
    sp = None


def _load_agentdistill_math_helpers() -> tuple[Any | None, Any | None]:
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


def extract_answer(text: Any) -> str | None:
    if text is None:
        return None
    s = str(text)
    if AD_EXTRACT_ANSWER is not None:
        try:
            extracted = AD_EXTRACT_ANSWER(s)
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
    nums = re.findall(r"-?\d*\.?\d+", s.replace(",", ""))
    if nums:
        return nums[-1]
    return None


def is_correct(pred: str | None, ref: str | None) -> bool:
    if pred is None or ref is None:
        return False
    if AD_MATH_EQUAL is not None:
        try:
            return bool(AD_MATH_EQUAL(pred, ref, timeout=True))
        except Exception:  # noqa: BLE001
            pass
    if pred.strip() == ref.strip():
        return True
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except Exception:
        pass
    if sp is not None:
        try:
            return sp.simplify(sp.sympify(pred.strip()) - sp.sympify(ref.strip())) == 0
        except Exception:  # noqa: BLE001
            pass
    return pred.strip() == ref.strip()


def load_qa_examples(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    data_path = cfg.get("data_path")
    if data_path:
        path = Path(data_path).expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if isinstance(data.get("examples"), list):
                examples = data["examples"]
            else:
                raise ValueError(f"Unsupported JSON object format in {path}")
        elif isinstance(data, list):
            examples = data
        else:
            raise ValueError(f"Unsupported dataset JSON type in {path}: {type(data).__name__}")

        limit = int(cfg.get("limit", 0))
        if limit > 0:
            examples = examples[:limit]
        return examples

    if "path" in cfg:
        ds = load_dataset(cfg["path"], cfg.get("config", "default"))
    else:
        raise ValueError("dataset.path is required")

    split = cfg.get("split")
    if split is not None and split not in ds:
        # Be resilient to dataset variants that do not expose the requested split.
        split = None

    if split is None:
        for candidate in ("test", "validation", "train"):
            if candidate in ds:
                split = candidate
                break
        if split is None:
            # Last resort: pick the first available split key.
            split = next(iter(ds.keys()), None)
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
