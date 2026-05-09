"""OpenAI LLM provider — thin wrapper over the OpenAI-compatible adapter."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleProvider
from ...config import OPENAI_API_KEY, OPENAI_MODEL_CATALOG


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API provider with streaming and automatic model fallback."""

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required when LLM_PROVIDER=openai"
            )
        super().__init__(
            name="openai",
            api_key=OPENAI_API_KEY,
            catalog=OPENAI_MODEL_CATALOG,
            supports_vision=True,
            supports_tools=True,
        )
