"""LLM provider factory and abstractions."""

from __future__ import annotations

import os

from .base import LLMProvider
from ...config import LLM_PROVIDER as _DEFAULT_LLM_PROVIDER


def _resolve_provider_name() -> str:
    """Pick the active provider, re-reading the env var so tests can flip it."""
    return os.getenv("LLM_PROVIDER", _DEFAULT_LLM_PROVIDER).strip().lower()


def get_llm_provider() -> LLMProvider:
    """Factory function to get the configured LLM provider."""
    name = _resolve_provider_name()
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if name == "local":
        from .local_provider import LocalProvider
        return LocalProvider()
    if name == "deepseek":
        from .deepseek_provider import DeepSeekProvider
        return DeepSeekProvider()
    if name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider()
    raise RuntimeError(
        f"Unknown LLM_PROVIDER='{name}'. "
        "Supported values: deepseek, openai, anthropic, local."
    )


__all__ = [
    "LLMProvider",
    "get_llm_provider",
]
