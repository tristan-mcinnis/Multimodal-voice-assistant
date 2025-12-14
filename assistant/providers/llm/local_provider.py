"""Local LLM provider implementation (LM Studio compatible)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import LLMProvider
from ...config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_MODEL,
    DEFAULT_MODEL_CATALOG,
)
from ...utils import log


class LocalProvider(LLMProvider):
    """Local LLM provider using OpenAI-compatible API (e.g., LM Studio)."""

    def __init__(self) -> None:
        self._client = OpenAI(base_url=LOCAL_LLM_BASE_URL, api_key=LOCAL_LLM_API_KEY)
        self._catalog = DEFAULT_MODEL_CATALOG
        log(f"Using local LLM at {LOCAL_LLM_BASE_URL}", title="LLM", style="bold blue")

    @property
    def client(self) -> OpenAI:
        return self._client

    @property
    def supports_vision(self) -> bool:
        # Vision not typically supported by local models
        return False

    @property
    def supports_tools(self) -> bool:
        # Tool support depends on the model, assume true for now
        return True

    def chat_completion(
        self,
        capability: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute chat completion using local model."""
        last_error: Optional[Exception] = None
        models = self._catalog.get(capability)

        # If no models configured for this capability, use default model
        if not models:
            models = [LOCAL_LLM_MODEL]

        for model_name in models:
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
