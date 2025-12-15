"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat_completion(
        self,
        capability: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute chat completion with automatic model fallback.

        Args:
            capability: The capability type ("conversation", "vision", "structured")
            messages: List of message dictionaries
            **kwargs: Additional arguments passed to the underlying API

        Returns:
            The API response object
        """
        pass

    def stream_chat_completion(
        self,
        capability: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute chat completion with streaming.

        Args:
            capability: The capability type ("conversation", "vision", "structured")
            messages: List of message dictionaries
            **kwargs: Additional arguments passed to the underlying API

        Returns:
            A generator yielding chunks of the response
        """
        raise NotImplementedError("Streaming not supported by this provider")

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider supports vision/image inputs."""
        pass

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider supports tool/function calling."""
        pass

    @property
    @abstractmethod
    def client(self) -> Any:
        """The underlying API client."""
        pass
