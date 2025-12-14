"""LLM provider factory and abstractions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import LLMProvider
from ...config import LLM_PROVIDER

if TYPE_CHECKING:
    pass


def get_llm_provider() -> LLMProvider:
    """Factory function to get the configured LLM provider."""
    if LLM_PROVIDER == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    elif LLM_PROVIDER == "local":
        from .local_provider import LocalProvider
        return LocalProvider()
    else:
        from .openai_provider import OpenAIProvider
        return OpenAIProvider()


__all__ = [
    "LLMProvider",
    "get_llm_provider",
]
