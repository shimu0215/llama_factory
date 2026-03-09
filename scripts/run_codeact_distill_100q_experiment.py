import argparse
import json
import os
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]

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
    parser = argparse.ArgumentParser(description="Run 100Q distillation effectiveness experiment.")
    parser.add_argument("--source-records-dir", required=True)
    parser.add_argument("--finetuned-qwen14-path", required=True)
    parser.add_argument("--work-dir", default="outputs/distill_100q_experiment")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-qwen14-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--base-qwen7-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--qwen14-infer-yaml", default="examples/inference/qwen3_14b_codeact.yaml")
    parser.add_argument("--qwen7-infer-yaml", default="examples/inference/qwen25_7b.yaml")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt-budget-tokens", type=int, default=16384)
    parser.add_argument("--recent-steps", type=int, default=2)
    parser.add_argument("--vllm-maxlen", type=int, default=24576)
    parser.add_argument("--vllm-gpu-util", type=float, default=0.25)
    parser.add_argument("--student-epochs", type=float, default=1.0)
    parser.add_argument("--student-save-steps", type=int, default=50)
    parser.add_argument("--student-grad-acc", type=int, default=16)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--api-ready-timeout", type=int, default=900)
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def resolve_model_path(path_str: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if path.is_dir() and (path / "config.json").exists():
        return str(path)

    if path.is_dir():
        checkpoint_dirs = sorted(
            [p for p in path.glob("checkpoint-*") if p.is_dir() and (p / "config.json").exists()],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        if checkpoint_dirs:
            picked = checkpoint_dirs[-1]
            print(f"[INFO] Using latest checkpoint as model path: {picked}")
            return str(picked)

    raise RuntimeError(
        f"Invalid model path: {path}. Expect a directory with config.json, "
        "or a parent directory containing checkpoint-*/config.json."
    )


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _tail_log(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return "(api log file not found)"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def wait_api_ready(proc: subprocess.Popen[Any], port: int, log_path: Path, timeout_sec: int = 900) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout_sec:
        ret = proc.poll()
        if ret is not None:
            tail = _tail_log(log_path)
            raise RuntimeError(
                f"API process exited early with code {ret}.\n"
                f"log: {log_path}\n"
                f"--- log tail ---\n{tail}"
            )
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    tail = _tail_log(log_path)
    raise RuntimeError(
        f"API not ready on port {port} within {timeout_sec}s.\n"
        f"log: {log_path}\n"
        f"--- log tail ---\n{tail}"
    )


def start_api(
    infer_yaml: str,
    model_name_or_path: str,
    log_path: Path,
    api_ready_timeout: int,
    vllm_maxlen: int,
    vllm_gpu_util: float,
    adapter_name_or_path: str | None = None,
) -> tuple[subprocess.Popen[Any], int]:
    port = pick_free_port()
    env = os.environ.copy()
    env["API_PORT"] = str(port)
    cmd = [
        "llamafactory-cli",
        "api",
        infer_yaml,
        f"model_name_or_path={model_name_or_path}",
        "infer_backend=vllm",
        "vllm_enforce_eager=true",
        f"vllm_maxlen={vllm_maxlen}",
        f"vllm_gpu_util={vllm_gpu_util}",
    ]
    if adapter_name_or_path:
        cmd.append(f"adapter_name_or_path={adapter_name_or_path}")
    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
    wait_api_ready(proc, port, log_path, timeout_sec=api_ready_timeout)
    return proc, port


def stop_api(proc: subprocess.Popen[Any]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def load_records_from_parts(source_dir: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(source_dir.glob("gsm_hard_codeact_records_part_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                qid = int(obj["question_id"])
                if qid not in records:
                    records[qid] = {
                        "question_id": qid,
                        "question": obj["question"],
                        "ground_truth": obj["ground_truth"],
                    }
    if not records:
        raise RuntimeError(f"No records found under {source_dir}")
    return records


def run_eval(
    *,
    name: str,
    model_name_or_path: str,
    infer_yaml: str,
    question_ids_file: Path,
    output_dir: Path,
    num_samples: int,
    temperature: float,
    max_steps: int,
    max_tokens: int,
    prompt_budget_tokens: int,
    recent_steps: int,
    api_ready_timeout: int,
    vllm_maxlen: int,
    vllm_gpu_util: float,
    adapter_name_or_path: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    api_log = output_dir / f"{name}_api.log"
    launch_attempts = [
        (vllm_maxlen, vllm_gpu_util),
        (min(vllm_maxlen, 16384), min(vllm_gpu_util, 0.22)),
        (12288, min(vllm_gpu_util, 0.18)),
    ]
    launch_error: Exception | None = None
    proc = None
    port = None
    for attempt_idx, (attempt_maxlen, attempt_mem_util) in enumerate(launch_attempts, start=1):
        try:
            print(
                f"[INFO] Launch API attempt {attempt_idx}: "
                f"vllm_maxlen={attempt_maxlen}, vllm_gpu_util={attempt_mem_util}"
            )
            proc, port = start_api(
                infer_yaml,
                model_name_or_path,
                api_log,
                api_ready_timeout,
                attempt_maxlen,
                attempt_mem_util,
                adapter_name_or_path,
            )
            break
        except Exception as e:  # noqa: BLE001
            launch_error = e
            print(f"[WARN] API launch attempt {attempt_idx} failed: {e}")
            time.sleep(3)

    if proc is None or port is None:
        assert launch_error is not None
        raise launch_error
    try:
        cmd = [
            "python",
            "scripts/run_gsm_hard_smolagents_codeact.py",
            "--base-url",
            f"http://127.0.0.1:{port}/v1",
            "--api-key",
            "0",
            "--model",
            "test",
            "--limit",
            "0",
            "--num-samples",
            str(num_samples),
            "--temperature",
            str(temperature),
            "--max-steps",
            str(max_steps),
            "--max-tokens",
            str(max_tokens),
            "--recent-steps",
            str(recent_steps),
            "--prompt-budget-tokens",
            str(prompt_budget_tokens),
            "--record-shard-size",
            "200",
            "--group-shard-size",
            "100",
            "--context-shard-size",
            "200",
            "--question-ids-file",
            str(question_ids_file),
            "--output-dir",
            str(output_dir),
        ]
        run_cmd(cmd, cwd=ROOT)
    finally:
        stop_api(proc)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(output_dir.glob("gsm_hard_codeact_grouped_part_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                grouped[int(item["question_id"])] = item["paths"]

    first_correct = 0
    any_correct = 0
    for qid, paths in grouped.items():
        if paths and paths[0].get("correct", False):
            first_correct += 1
        if any(p.get("correct", False) for p in paths):
            any_correct += 1

    total = len(grouped)
    return {
        "name": name,
        "output_dir": str(output_dir),
        "total_questions": total,
        "first_sample_accuracy": first_correct / total if total else 0.0,
        "any_sample_accuracy": any_correct / total if total else 0.0,
        "grouped": grouped,
    }


def build_distill_sharegpt(
    *,
    output_jsonl: Path,
    teacher_grouped: dict[int, list[dict[str, Any]]],
    selected_qids: list[int],
    seed: int,
) -> int:
    rng = random.Random(seed)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_jsonl.open("w", encoding="utf-8") as out:
        for qid in selected_qids:
            paths = teacher_grouped.get(qid, [])
            correct_paths = [p for p in paths if p.get("correct", False)]
            if not correct_paths:
                continue
            pick = rng.choice(correct_paths)
            sample = {
                "id": f"distill_q{qid}",
                "messages": [
                    {"role": "system", "content": CODEACT_SYSTEM_PROMPT},
                    {"role": "user", "content": str(pick["question"])},
                    {"role": "assistant", "content": str(pick.get("cot_text", pick.get("final_answer", "")))},
                ],
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def write_dataset_info(dataset_dir: Path, entries: dict[str, str]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, file_name in entries.items():
        info[name] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    (dataset_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def write_student_yaml(
    yaml_path: Path,
    *,
    dataset_name: str,
    dataset_dir: Path,
    output_dir: Path,
    base_qwen7_model: str,
    teacher_ref_model: str,
    epochs: float,
    save_steps: int,
    grad_acc: int,
    lr: float,
) -> None:
    content = f"""### model
model_name_or_path: {base_qwen7_model}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
use_asft_loss: true
asft_alpha: 0.1
ref_model: {teacher_ref_model}

### dataset
dataset: {dataset_name}
dataset_dir: {dataset_dir}
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 1
dataloader_num_workers: 4

### output
output_dir: {output_dir}
logging_steps: 10
save_strategy: steps
save_steps: {save_steps}
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: {grad_acc}
learning_rate: {lr}
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
ddp_timeout: 180000000
"""
    yaml_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    source_map = load_records_from_parts(Path(args.source_records_dir).expanduser().resolve())
    finetuned_qwen14_path = resolve_model_path(args.finetuned_qwen14_path)
    all_qids = sorted(source_map.keys())
    if len(all_qids) < args.sample_size:
        raise RuntimeError(f"Only {len(all_qids)} unique questions found, < sample-size {args.sample_size}.")

    rng = random.Random(args.seed)
    sampled_qids = sorted(rng.sample(all_qids, args.sample_size))
    question_ids_file = work_dir / "sampled_question_ids.json"
    question_ids_file.write_text(json.dumps(sampled_qids, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_root = work_dir / "evals"
    base14_eval = run_eval(
        name="base14",
        model_name_or_path=args.base_qwen14_model,
        infer_yaml=args.qwen14_infer_yaml,
        question_ids_file=question_ids_file,
        output_dir=eval_root / "base14_single",
        num_samples=1,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        prompt_budget_tokens=args.prompt_budget_tokens,
        recent_steps=args.recent_steps,
        api_ready_timeout=args.api_ready_timeout,
        vllm_maxlen=args.vllm_maxlen,
        vllm_gpu_util=args.vllm_gpu_util,
    )
    ft14_eval = run_eval(
        name="ft14",
        model_name_or_path=finetuned_qwen14_path,
        infer_yaml=args.qwen14_infer_yaml,
        question_ids_file=question_ids_file,
        output_dir=eval_root / "ft14_triple",
        num_samples=3,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        prompt_budget_tokens=args.prompt_budget_tokens,
        recent_steps=args.recent_steps,
        api_ready_timeout=args.api_ready_timeout,
        vllm_maxlen=args.vllm_maxlen,
        vllm_gpu_util=args.vllm_gpu_util,
    )
    base7_eval = run_eval(
        name="base7",
        model_name_or_path=args.base_qwen7_model,
        infer_yaml=args.qwen7_infer_yaml,
        question_ids_file=question_ids_file,
        output_dir=eval_root / "base7_single",
        num_samples=1,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        prompt_budget_tokens=args.prompt_budget_tokens,
        recent_steps=args.recent_steps,
        api_ready_timeout=args.api_ready_timeout,
        vllm_maxlen=args.vllm_maxlen,
        vllm_gpu_util=args.vllm_gpu_util,
    )

    both_correct_qids: list[int] = []
    for qid in sampled_qids:
        b14_ok = any(p.get("correct", False) for p in base14_eval["grouped"].get(qid, []))
        f14_ok = any(p.get("correct", False) for p in ft14_eval["grouped"].get(qid, []))
        if b14_ok and f14_ok:
            both_correct_qids.append(qid)
    (work_dir / "distill_overlap_qids.json").write_text(
        json.dumps(both_correct_qids, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    distill_dir = work_dir / "distill_data"
    base14_dataset_name = "distill_base14_teacher_100q"
    ft14_dataset_name = "distill_ft14_teacher_100q"
    base14_jsonl = distill_dir / "distill_base14_teacher_100q.jsonl"
    ft14_jsonl = distill_dir / "distill_ft14_teacher_100q.jsonl"
    kept_base14 = build_distill_sharegpt(
        output_jsonl=base14_jsonl,
        teacher_grouped=base14_eval["grouped"],
        selected_qids=both_correct_qids,
        seed=args.seed,
    )
    kept_ft14 = build_distill_sharegpt(
        output_jsonl=ft14_jsonl,
        teacher_grouped=ft14_eval["grouped"],
        selected_qids=both_correct_qids,
        seed=args.seed + 1,
    )
    write_dataset_info(
        distill_dir,
        {
            base14_dataset_name: base14_jsonl.name,
            ft14_dataset_name: ft14_jsonl.name,
        },
    )

    train_root = work_dir / "student_runs"
    base14_student_out = train_root / "qwen7_from_base14"
    ft14_student_out = train_root / "qwen7_from_ft14"
    base14_yaml = work_dir / "train_qwen7_from_base14.yaml"
    ft14_yaml = work_dir / "train_qwen7_from_ft14.yaml"
    write_student_yaml(
        base14_yaml,
        dataset_name=base14_dataset_name,
        dataset_dir=distill_dir,
        output_dir=base14_student_out,
        base_qwen7_model=args.base_qwen7_model,
        teacher_ref_model=args.base_qwen14_model,
        epochs=args.student_epochs,
        save_steps=args.student_save_steps,
        grad_acc=args.student_grad_acc,
        lr=args.student_lr,
    )
    write_student_yaml(
        ft14_yaml,
        dataset_name=ft14_dataset_name,
        dataset_dir=distill_dir,
        output_dir=ft14_student_out,
        base_qwen7_model=args.base_qwen7_model,
        teacher_ref_model=finetuned_qwen14_path,
        epochs=args.student_epochs,
        save_steps=args.student_save_steps,
        grad_acc=args.student_grad_acc,
        lr=args.student_lr,
    )

    run_cmd(["llamafactory-cli", "train", str(base14_yaml)], cwd=ROOT)
    run_cmd(["llamafactory-cli", "train", str(ft14_yaml)], cwd=ROOT)

    student_from_base14_eval = run_eval(
        name="student_from_base14",
        model_name_or_path=args.base_qwen7_model,
        infer_yaml=args.qwen7_infer_yaml,
        question_ids_file=question_ids_file,
        output_dir=eval_root / "student_from_base14_single",
        num_samples=1,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        prompt_budget_tokens=args.prompt_budget_tokens,
        recent_steps=args.recent_steps,
        api_ready_timeout=args.api_ready_timeout,
        vllm_maxlen=args.vllm_maxlen,
        vllm_gpu_util=args.vllm_gpu_util,
        adapter_name_or_path=str(base14_student_out),
    )
    student_from_ft14_eval = run_eval(
        name="student_from_ft14",
        model_name_or_path=args.base_qwen7_model,
        infer_yaml=args.qwen7_infer_yaml,
        question_ids_file=question_ids_file,
        output_dir=eval_root / "student_from_ft14_single",
        num_samples=1,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        prompt_budget_tokens=args.prompt_budget_tokens,
        recent_steps=args.recent_steps,
        api_ready_timeout=args.api_ready_timeout,
        vllm_maxlen=args.vllm_maxlen,
        vllm_gpu_util=args.vllm_gpu_util,
        adapter_name_or_path=str(ft14_student_out),
    )

    result = {
        "sampled_question_count": len(sampled_qids),
        "sampled_question_ids_file": str(question_ids_file),
        "base14_eval": {k: v for k, v in base14_eval.items() if k != "grouped"},
        "ft14_eval": {k: v for k, v in ft14_eval.items() if k != "grouped"},
        "base7_eval": {k: v for k, v in base7_eval.items() if k != "grouped"},
        "overlap_question_count": len(both_correct_qids),
        "overlap_qids_file": str(work_dir / "distill_overlap_qids.json"),
        "kept_distill_samples_base14": kept_base14,
        "kept_distill_samples_ft14": kept_ft14,
        "student_from_base14_eval": {k: v for k, v in student_from_base14_eval.items() if k != "grouped"},
        "student_from_ft14_eval": {k: v for k, v in student_from_ft14_eval.items() if k != "grouped"},
    }
    result_path = work_dir / "experiment_summary.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(result_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
