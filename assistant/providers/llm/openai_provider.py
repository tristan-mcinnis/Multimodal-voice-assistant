"""OpenAI LLM provider implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import LLMProvider
from ...config import OPENAI_API_KEY, DEFAULT_MODEL_CATALOG, LLM_PROVIDER
from ...utils import log


class OpenAIProvider(LLMProvider):
    """OpenAI API provider with automatic model fallback."""

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY environment variable is required when LLM_PROVIDER=openai")
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._catalog = DEFAULT_MODEL_CATALOG

    @property
    def client(self) -> OpenAI:
        return self._client

    @property
    def supports_vision(self) -> bool:
        return bool(self._catalog.get("vision"))

    @property
    def supports_tools(self) -> bool:
        return True

    def chat_completion(
        self,
        capability: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute chat completion with automatic model fallback."""
        last_error: Optional[Exception] = None
        for model_name in self._catalog.get(capability):
            try:
                response = self._client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **kwargs,
                )
                log(
                    f"Model '{model_name}' served the '{capability}' request.",
                    title="MODEL",
                    style="bold blue",
                )
                return response
            except Exception as exc:  # noqa: BLE001 - surface fallback errors
                last_error = exc
                log(
                    f"Model '{model_name}' unavailable ({exc}). Trying fallback...",
                    title="MODEL",
                    style="bold yellow",
                )
        raise RuntimeError(f"All models failed for capability '{capability}'.") from last_error
