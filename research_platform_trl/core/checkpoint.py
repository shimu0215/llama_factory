import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageState:
    completed: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class StageCheckpoint:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = StageState()
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        obj = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.state = StageState(
            completed=obj.get("completed", {}),
            metadata=obj.get("metadata", {}),
        )

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "completed": self.state.completed,
                    "metadata": self.state.metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def done(self, stage_name: str) -> bool:
        return bool(self.state.completed.get(stage_name, False))

    def mark_done(self, stage_name: str) -> None:
        self.state.completed[stage_name] = True
        self.save()

    def set_meta(self, key: str, value: Any) -> None:
        self.state.metadata[key] = value
        self.save()

    def get_meta(self, key: str, default: Any = None) -> Any:
        return self.state.metadata.get(key, default)
