from typing import Any


def create_smolagents_model(model_cfg: dict[str, Any]) -> Any:
    backend = model_cfg.get("backend", "openai_compatible")

    if backend == "openai_compatible":
        from smolagents import OpenAIModel

        return OpenAIModel(
            model_id=model_cfg["model_id"],
            api_base=model_cfg["api_base"],
            api_key=model_cfg.get("api_key", "0"),
            temperature=float(model_cfg.get("temperature", 0.7)),
            max_tokens=int(model_cfg.get("max_tokens", 1024)),
        )

    if backend == "transformers_local":
        # Requires smolagents transformers model integration.
        from smolagents import TransformersModel

        return TransformersModel(
            model_id=model_cfg["model_id"],
            max_new_tokens=int(model_cfg.get("max_tokens", 1024)),
            temperature=float(model_cfg.get("temperature", 0.7)),
            device_map=model_cfg.get("device_map", "auto"),
        )

    raise ValueError(f"Unsupported model backend: {backend}")
