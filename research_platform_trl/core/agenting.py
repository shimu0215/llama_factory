from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Optional

from smolagents import CodeAgent
from smolagents.agents import MultiStepAgent
from smolagents.monitoring import LogLevel


@dataclass
class CompressionConfig:
    enabled: bool = True
    mode: str = "auto"  # auto|fixed
    model_max_context_tokens: int | None = None
    prompt_budget_tokens: int | None = None
    recent_steps: int = 2
    max_summary_chars: int = 1200
    max_observation_chars: int = 1500
    max_step_chars: int = 1200


class RollingMemoryCodeAgent(CodeAgent):
    def __init__(self, *args: Any, compression: CompressionConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.comp = compression
        self.visible_contexts: list[dict[str, Any]] = []

    def _effective_budget(self) -> int | None:
        if not self.comp.enabled:
            return None
        if self.comp.prompt_budget_tokens:
            return int(self.comp.prompt_budget_tokens)
        if self.comp.mode == "auto" and self.comp.model_max_context_tokens:
            return int(self.comp.model_max_context_tokens * 0.85)
        return None

    def _message_role(self, message: Any) -> Any:
        if isinstance(message, dict):
            return message.get("role")
        return getattr(message, "role", None)

    def _message_content(self, message: Any) -> Any:
        if isinstance(message, dict):
            return message.get("content", [])
        return getattr(message, "content", [])

    def _replace_content(self, message: Any, content: Any) -> Any:
        if isinstance(message, dict):
            return {**message, "content": content}
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"content": content})
        message.content = content
        return message

    def _serialize_message(self, message: Any) -> dict[str, Any]:
        content = self._message_content(message)
        if isinstance(content, str):
            serial = content
        else:
            serial = []
            for item in content:
                serial.append(item if isinstance(item, dict) else str(item))
        return {"role": str(self._message_role(message)), "content": serial}

    def _estimate_tokens(self, messages: list[Any]) -> int:
        chars = 0
        for m in messages:
            content = self._message_content(m)
            if isinstance(content, str):
                chars += len(content)
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chars += len(item.get("text", ""))
        return chars // 4

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        keep = max(32, max_chars // 2)
        return f"{text[:keep]}\n...[truncated]...\n{text[-keep:]}"

    def _truncate_messages(self, messages: list[Any], step_chars: int, obs_chars: int) -> list[Any]:
        out: list[Any] = []
        for m in messages:
            content = self._message_content(m)
            if isinstance(content, str):
                out.append(self._replace_content(m, self._truncate_text(content, step_chars)))
                continue
            new_content = []
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    new_content.append(item)
                    continue
                t = item.get("text", "")
                limit = obs_chars if "Observation:" in t else step_chars
                new_content.append({**item, "text": self._truncate_text(t, limit)})
            out.append(self._replace_content(m, new_content))
        return out

    def _summarize_old_steps(self, steps: list[Any], max_chars: int) -> str:
        if not steps:
            return ""
        lines = ["Previous progress summary:"]
        for i, step in enumerate(steps, start=1):
            model_output = getattr(step, "model_output", None)
            obs = getattr(step, "observations", None)
            err = getattr(step, "error", None)
            parts = []
            if model_output:
                parts.append(f"reasoning={self._truncate_text(str(model_output), 160)}")
            if obs:
                parts.append(f"observation={self._truncate_text(str(obs), 200)}")
            if err:
                parts.append(f"error={self._truncate_text(str(err), 120)}")
            if parts:
                lines.append(f"- Step {i}: " + "; ".join(parts))
        return self._truncate_text("\n".join(lines), max_chars)

    def _build_compacted_messages(self, recent_steps: int, summary_chars: int, obs_chars: int, step_chars: int) -> list[Any]:
        messages = self.memory.system_prompt.to_messages(summary_mode=False)
        steps = list(self.memory.steps)
        if not steps:
            return messages

        first = steps[0]
        if hasattr(first, "task"):
            messages.extend(first.to_messages(summary_mode=False))
            action_steps = steps[1:]
        else:
            action_steps = steps

        rs = max(1, min(recent_steps, len(action_steps))) if action_steps else 0
        old_steps = action_steps[:-rs] if rs else action_steps
        new_steps = action_steps[-rs:] if rs else []

        summary = self._summarize_old_steps(old_steps, summary_chars)
        if summary:
            messages.append({"role": "user", "content": [{"type": "text", "text": summary}]})

        for step in new_steps:
            messages.extend(step.to_messages(summary_mode=False))

        return self._truncate_messages(messages, step_chars, obs_chars)

    def write_memory_to_messages(self, summary_mode: Optional[bool] = False) -> list[Any]:
        if summary_mode:
            return super().write_memory_to_messages(summary_mode=True)

        full = super().write_memory_to_messages(summary_mode=False)
        budget = self._effective_budget()
        full_tokens = self._estimate_tokens(full)
        if budget is None or full_tokens <= budget:
            self.visible_contexts.append(
                {
                    "call_index": len(self.visible_contexts),
                    "compressed": False,
                    "estimated_tokens_before": full_tokens,
                    "estimated_tokens_after": full_tokens,
                    "budget": budget,
                    "messages": [self._serialize_message(m) for m in full],
                }
            )
            return full

        recent = self.comp.recent_steps
        s_chars = self.comp.max_summary_chars
        o_chars = self.comp.max_observation_chars
        t_chars = self.comp.max_step_chars
        compacted = self._build_compacted_messages(recent, s_chars, o_chars, t_chars)

        while self._estimate_tokens(compacted) > budget:
            changed = False
            if s_chars > 300:
                s_chars = max(300, s_chars // 2)
                changed = True
            elif o_chars > 400:
                o_chars = max(400, o_chars // 2)
                changed = True
            elif t_chars > 400:
                t_chars = max(400, t_chars // 2)
                changed = True
            elif recent > 1:
                recent -= 1
                changed = True
            if not changed:
                break
            compacted = self._build_compacted_messages(recent, s_chars, o_chars, t_chars)

        self.visible_contexts.append(
            {
                "call_index": len(self.visible_contexts),
                "compressed": True,
                "estimated_tokens_before": full_tokens,
                "estimated_tokens_after": self._estimate_tokens(compacted),
                "budget": budget,
                "recent_steps_used": recent,
                "summary_chars_used": s_chars,
                "observation_chars_used": o_chars,
                "step_chars_used": t_chars,
                "messages": [self._serialize_message(m) for m in compacted],
            }
        )
        return compacted


def create_codeact_agent(
    *,
    model_client: Any,
    system_prompt: str,
    tools: list[Any],
    max_steps: int,
    compression_cfg: CompressionConfig,
    enable_rolling_memory: bool = True,
    code_block_tags: str | tuple[str, str] | None = None,
) -> CodeAgent:
    common_kwargs = {
        "tools": tools,
        "model": model_client,
        "additional_authorized_imports": ["numpy", "sympy", "numpy.linalg"],
        "max_steps": max_steps,
        "verbosity_level": LogLevel.ERROR,
        "set_timeout": True,
        "code_block_tags": code_block_tags,
    }
    if system_prompt:
        common_kwargs["instructions"] = system_prompt
    allowed = set(inspect.signature(CodeAgent.__init__).parameters) | set(
        inspect.signature(MultiStepAgent.__init__).parameters
    )
    common_kwargs = {k: v for k, v in common_kwargs.items() if k in allowed}
    if enable_rolling_memory:
        return RollingMemoryCodeAgent(
            **common_kwargs,
            compression=compression_cfg,
        )
    return CodeAgent(**common_kwargs)
