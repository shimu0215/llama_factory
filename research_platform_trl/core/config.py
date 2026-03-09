from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoadedConfig:
    path: Path
    raw: dict[str, Any]


def load_yaml_config(path: str | Path) -> LoadedConfig:
    cfg_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return LoadedConfig(path=cfg_path, raw=raw)


def get_required(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: {key}")
    return cfg[key]
