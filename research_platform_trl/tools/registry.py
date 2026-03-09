from typing import Any


def create_tools(tool_cfg: dict[str, Any]) -> list[Any]:
    names = tool_cfg.get("enabled", ["python_exec"])
    tools: list[Any] = []

    # For CodeAgent, Python execution is built in via code blocks, so python_exec is implicit.
    # We keep this registry for future extensions like web search.
    for name in names:
        if name == "python_exec":
            continue
        if name == "duckduckgo_search":
            from smolagents import DuckDuckGoSearchTool

            tools.append(DuckDuckGoSearchTool())
            continue
        raise ValueError(f"Unknown tool name: {name}")
    return tools
