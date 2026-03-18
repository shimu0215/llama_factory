import argparse
import subprocess
from datetime import datetime
from pathlib import Path


ACTIVE_STATES = {"RUNNING", "COMPLETING", "CONFIGURING"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check dual hold-job reservations and cancel the unneeded one.")
    parser.add_argument("--gpuq-job", required=True)
    parser.add_argument("--contrib-job", required=True)
    parser.add_argument(
        "--log-file",
        default="/scratch/wzhao20/llama_factory/hpc-results/dual_hold_job_checks.log",
    )
    return parser.parse_args()


def job_state(job_id: str) -> str:
    cmd = [
        "sacct",
        "-j",
        job_id,
        "--format=JobID,State",
        "-n",
        "-P",
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    states: list[str] = []
    for line in res.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2 and parts[0] == job_id:
            states.append(parts[1].strip())
    if states:
        return states[0]

    cmd = ["squeue", "-j", job_id, "-h", "-o", "%T"]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    state = res.stdout.strip()
    return state or "UNKNOWN"


def cancel_job(job_id: str) -> str:
    res = subprocess.run(["scancel", job_id], check=False, capture_output=True, text=True)
    if res.returncode == 0:
        return f"cancel:{job_id}"
    return f"cancel_failed:{job_id}"


def append_log(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file)

    gpuq_state = job_state(args.gpuq_job)
    contrib_state = job_state(args.contrib_job)

    gpuq_active = gpuq_state in ACTIVE_STATES
    contrib_active = contrib_state in ACTIVE_STATES

    action = "none"
    if gpuq_active and contrib_active:
        action = cancel_job(args.contrib_job)
    elif gpuq_active:
        action = cancel_job(args.contrib_job)
    elif contrib_active:
        action = cancel_job(args.gpuq_job)

    timestamp = datetime.now().isoformat(timespec="seconds")
    append_log(
        log_file,
        f"{timestamp} gpuq={args.gpuq_job}:{gpuq_state} contrib={args.contrib_job}:{contrib_state} action={action}",
    )
    print(
        f"gpuq={args.gpuq_job}:{gpuq_state} contrib={args.contrib_job}:{contrib_state} action={action}"
    )


if __name__ == "__main__":
    main()
