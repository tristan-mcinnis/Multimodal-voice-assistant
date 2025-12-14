"""Local LLM provider implementation (LM Studio compatible)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

from .base import LLMProvider
from ...config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_MODEL,
    ModelCatalog,
)
from ...utils import log


class LocalProvider(LLMProvider):
    """Local LLM provider using OpenAI-compatible API (e.g., LM Studio)."""

    def __init__(self) -> None:
        # Use httpx client with explicit transports to bypass system proxies for localhost
        http_client = httpx.Client(mounts={
            "http://localhost": httpx.HTTPTransport(),
            "http://127.0.0.1": httpx.HTTPTransport(),
        })
        self._client = OpenAI(base_url=LOCAL_LLM_BASE_URL, api_key=LOCAL_LLM_API_KEY, http_client=http_client)
        self._detected_model = self._detect_model()
        # Build catalog with detected model
        model_name = self._detected_model or LOCAL_LLM_MODEL
        self._catalog = ModelCatalog({
            "conversation": [model_name],
            "vision": [],
            "structured": [model_name],
        })
        log(f"Using local LLM at {LOCAL_LLM_BASE_URL}", title="LLM", style="bold blue")
        if self._detected_model:
            log(f"Auto-detected model: {self._detected_model}", title="LLM", style="bold blue")

    def _detect_model(self) -> Optional[str]:
        """Auto-detect the first available chat model from LM Studio."""
        import time
        if LOCAL_LLM_MODEL != "local-model":
            # User specified a model, use it
            return LOCAL_LLM_MODEL
        # Retry up to 3 times with a short delay
        for attempt in range(3):
            try:
                models = self._client.models.list()
                # Filter out embedding models and pick the first chat model
                for model in models.data:
                    model_id = model.id.lower()
                    if "embed" not in model_id and "ocr" not in model_id:
                        return model.id
                # Fallback to first model if no chat model found
                if models.data:
                    return models.data[0].id
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)  # Wait 1 second before retry
                else:
                    log(f"Model detection failed after 3 attempts: {e}", title="LLM", style="bold yellow")
        return None

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

        # If no models configured for this capability, use detected or default model
        if not models:
            models = [self._detected_model or LOCAL_LLM_MODEL]

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
