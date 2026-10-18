"""DeepSeek configuration over its documented OpenAI-compatible endpoint."""

from typing import Any

from app.llm.providers.openai_compatible import OpenAICompatibleLLM


class DeepSeekLLM(OpenAICompatibleLLM):
    """Use DeepSeek Chat Completions JSON mode plus local Pydantic validation."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["structured_output_mode"] = "json_object"
        super().__init__(**kwargs)
