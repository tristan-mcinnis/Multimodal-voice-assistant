"""Unified tool-call loop.

One module owns "how do the LLM and the tool registry talk to each other."
Both streaming and non-streaming callers go through `ToolLoop.stream`; the
non-streaming path is a thin collect-from-stream wrapper.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..providers.llm import LLMProvider
from ..utils import log
from .registry import ToolRegistry


def _empty_call() -> dict[str, Any]:
    return {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}


def _accumulate_stream(stream: Any) -> tuple[str, list[dict[str, Any]]]:
    """Single pass over a chat-completion stream — collect text, return tool calls.

    Yields nothing; the caller wanting live chunks should use `_iter_stream`.
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                idx = tc.index if tc.index is not None else 0
                while len(tool_calls) <= idx:
                    tool_calls.append(_empty_call())
                target = tool_calls[idx]
                if tc.id:
                    target["id"] += tc.id
                if tc.function:
                    if tc.function.name:
                        target["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        target["function"]["arguments"] += tc.function.arguments
        if getattr(delta, "content", None):
            text_parts.append(delta.content)
    return "".join(text_parts), tool_calls


class ToolLoop:
    """Run the LLM↔tools conversation loop, yielding text chunks as they stream.

    Mutates `messages` in place to record the full exchange, mirroring the
    conventions used by both Claude and OpenAI clients (assistant turn with
    tool_calls, then one `tool` message per call, then iterate).
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        capability: str = "conversation",
        tools_enabled: bool = True,
    ) -> None:
        self._llm = llm_provider
        self._tools = tool_registry
        self._capability = capability
        self._tools_enabled = tools_enabled

    def _request_params(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        params: dict[str, Any] = {"messages": messages}
        if self._tools_enabled and self._tools.has_tools():
            params["tools"] = self._tools.as_openai_tools()
            params["tool_choice"] = "auto"
        return params

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        """Run the loop, yielding assistant text chunks live."""
        while True:
            params = self._request_params(messages)
            try:
                stream = self._llm.stream_chat_completion(self._capability, **params)
            except NotImplementedError:
                log(
                    "Provider does not support streaming; collecting full response.",
                    title="LLM",
                    style="yellow",
                )
                yield from self._fallback_complete(messages)
                return

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for chunk in stream:
                delta = chunk.choices[0].delta
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        while len(tool_calls) <= idx:
                            tool_calls.append(_empty_call())
                        target = tool_calls[idx]
                        if tc.id:
                            target["id"] += tc.id
                        if tc.function:
                            if tc.function.name:
                                target["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                target["function"]["arguments"] += tc.function.arguments
                if getattr(delta, "content", None):
                    text_parts.append(delta.content)
                    yield delta.content

            if tool_calls:
                self._record_tool_turn(messages, tool_calls)
                self._execute_tool_calls(messages, tool_calls)
                continue

            messages.append({"role": "assistant", "content": "".join(text_parts)})
            return

    def complete(self, messages: list[dict[str, Any]]) -> str:
        """Run the loop synchronously, returning the final text."""
        return "".join(self.stream(messages))

    def _record_tool_turn(
        self,
        messages: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
    ) -> None:
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

    def _execute_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
    ) -> None:
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments", "{}")
            log(f"Executing tool: {name}", title="TOOL", style="bold magenta")
            result = self._tools.execute(name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": result,
                }
            )

    def _fallback_complete(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        """Non-streaming fallback for providers that lack stream support."""
        from ..utils import extract_message_text, iter_tool_calls, message_to_dict

        while True:
            params = self._request_params(messages)
            response = self._llm.chat_completion(self._capability, **params)
            message_dict = message_to_dict(response.choices[0].message)
            tool_calls = iter_tool_calls(message_dict) if self._tools_enabled else []

            if tool_calls:
                messages.append(message_dict)
                self._execute_tool_calls(messages, tool_calls)
                continue

            messages.append(message_dict)
            yield extract_message_text(message_dict)
            return
