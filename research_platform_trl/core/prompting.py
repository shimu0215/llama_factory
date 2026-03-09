from pathlib import Path
from typing import Any

import yaml


def load_codeact_prompt(prompt_file: str | Path) -> str:
    path = Path(prompt_file).expanduser().resolve()
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "system_prompt" in data:
        return str(data["system_prompt"])
    if isinstance(data, str):
        return data
    raise ValueError(f"Invalid prompt file format: {path}")
