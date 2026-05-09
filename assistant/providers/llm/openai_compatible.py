"""Shared adapter for any OpenAI-Chat-Completions-compatible service.

DeepSeek, LM Studio, and OpenAI itself all speak the same wire format. One
adapter satisfies the LLMProvider seam for all of them — concrete providers
become thin factories that supply (name, base_url, api_key, catalog).
"""

from __future__ import annotations

from typing import Any

import httpx
from openai import OpenAI

from ...config import ModelCatalog
from ...utils import log
from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider for any service that speaks OpenAI's Chat Completions API."""

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        catalog: ModelCatalog,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        supports_vision: bool = False,
        supports_tools: bool = True,
    ) -> None:
        self._name = name
        self._catalog = catalog
        self._supports_vision = supports_vision
        self._supports_tools = supports_tools

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self._client = OpenAI(**client_kwargs)

    @property
    def name(self) -> str:
        return self._name

    @property
    def client(self) -> OpenAI:
        return self._client

    @property
    def catalog(self) -> ModelCatalog:
        return self._catalog

    @property
    def supports_vision(self) -> bool:
        return self._supports_vision and bool(self._catalog.get("vision"))

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    def _models_for(self, capability: str) -> list[str]:
        models = self._catalog.get(capability)
        if models:
            return models
        # Fall back to conversation models if the capability list is empty.
        return self._catalog.get("conversation")

    def _try_each(self, capability: str, **call_kwargs: Any) -> Any:
        last_error: Exception | None = None
        for model_name in self._models_for(capability):
            try:
                response = self._client.chat.completions.create(model=model_name, **call_kwargs)
                log(
                    f"{self._name}: '{model_name}' served the '{capability}' request.",
                    title="MODEL",
                    style="bold blue",
                )
                return response
            except Exception as exc:  # noqa: BLE001 - surface fallback errors
                last_error = exc
                log(
                    f"{self._name}: '{model_name}' unavailable ({exc}). Trying fallback...",
                    title="MODEL",
                    style="bold yellow",
                )
        raise RuntimeError(
            f"{self._name}: all models failed for capability '{capability}'."
        ) from last_error

    def chat_completion(
        self,
        capability: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("stream", None)
        return self._try_each(capability, messages=messages, **kwargs)

    def stream_chat_completion(
        self,
        capability: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        kwargs["stream"] = True
        return self._try_each(capability, messages=messages, **kwargs)
