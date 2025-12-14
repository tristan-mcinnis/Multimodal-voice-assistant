"""Anthropic (Claude) LLM provider implementation.

This is a stub implementation ready for future Claude integration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import LLMProvider
from ...config import (
    ANTHROPIC_API_KEY,
    CLAUDE_PREFERRED_CHAT_MODEL,
    CLAUDE_PREFERRED_VISION_MODEL,
    CLAUDE_PREFERRED_TOOL_MODEL,
    DEFAULT_MODEL_CATALOG,
)
from ...utils import log


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider.

    This is a stub implementation. To complete the integration:
    1. pip install anthropic
    2. Implement the chat_completion method using the Anthropic SDK
    3. Handle Claude's different message format for tool calling
    """

    def __init__(self) -> None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is required when LLM_PROVIDER=anthropic"
            )

        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
            log("Initialized Anthropic Claude provider", title="LLM", style="bold blue")
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        self._catalog = DEFAULT_MODEL_CATALOG

    @property
    def client(self) -> Any:
        return self._client

    @property
    def supports_vision(self) -> bool:
        return True  # Claude supports vision

    @property
    def supports_tools(self) -> bool:
        return True  # Claude supports tool use

    def chat_completion(
        self,
        capability: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute chat completion using Claude.

        Note: Claude's API has a different format than OpenAI's.
        This method converts OpenAI-style messages to Claude's format.
        """
        last_error: Optional[Exception] = None
        models = self._catalog.get(capability)

        if not models:
            models = [CLAUDE_PREFERRED_CHAT_MODEL]

        # Extract system message if present
        system_message = None
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                user_messages.append(msg)

        for model_name in models:
            try:
                # Build Claude API request
                request_kwargs = {
                    "model": model_name,
                    "messages": self._convert_messages(user_messages),
                    "max_tokens": kwargs.get("max_tokens", 4096),
                }

                if system_message:
                    request_kwargs["system"] = system_message

                # Handle tools if provided
                if "tools" in kwargs:
                    request_kwargs["tools"] = self._convert_tools(kwargs["tools"])

                response = self._client.messages.create(**request_kwargs)

                log(
                    f"Model '{model_name}' served the '{capability}' request.",
                    title="MODEL",
                    style="bold blue",
                )

                # Wrap response in OpenAI-compatible format for compatibility
                return self._wrap_response(response)

            except Exception as exc:  # noqa: BLE001 - surface fallback errors
                last_error = exc
                log(
                    f"Model '{model_name}' unavailable ({exc}). Trying fallback...",
                    title="MODEL",
                    style="bold yellow",
                )

        raise RuntimeError(f"All models failed for capability '{capability}'.") from last_error

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Claude format."""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map OpenAI roles to Claude roles
            if role == "assistant":
                role = "assistant"
            elif role == "tool":
                # Claude handles tool results differently
                role = "user"
                content = f"Tool result: {content}"
            else:
                role = "user"

            converted.append({"role": role, "content": content})
        return converted

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Claude format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                converted.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
        return converted

    def _wrap_response(self, response: Any) -> Any:
        """Wrap Claude response in an OpenAI-compatible format.

        This allows the rest of the codebase to work with Claude responses
        using the same interface as OpenAI responses.
        """
        # Create a simple wrapper class that mimics OpenAI response structure
        class WrappedMessage:
            def __init__(self, content: str, tool_calls: Optional[List] = None):
                self.content = content
                self.tool_calls = tool_calls or []

            def model_dump(self) -> Dict[str, Any]:
                return {
                    "role": "assistant",
                    "content": self.content,
                    "tool_calls": self.tool_calls,
                }

        class WrappedChoice:
            def __init__(self, message: WrappedMessage):
                self.message = message

        class WrappedResponse:
            def __init__(self, content: str, tool_calls: Optional[List] = None):
                self.choices = [WrappedChoice(WrappedMessage(content, tool_calls))]

        # Extract content from Claude response
        content_blocks = response.content if hasattr(response, "content") else []
        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if hasattr(block, "type"):
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": str(block.input),
                        },
                    })

        return WrappedResponse(text_content, tool_calls if tool_calls else None)
