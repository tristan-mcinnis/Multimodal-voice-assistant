"""Tests for the unified ToolLoop.

Drives the loop with a fake LLM provider that scripts a sequence of
streamed responses. Verifies the loop:
  - yields text chunks live
  - dispatches tool calls and appends `tool` messages
  - iterates until the model returns a clean reply
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

from assistant.providers.llm.base import LLMProvider
from assistant.tools.loop import ToolLoop
from assistant.tools.registry import ToolRegistry


def _text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))]
    )


def _tool_chunk(idx: int, *, call_id: str = "", name: str = "", args: str = "") -> SimpleNamespace:
    fn = SimpleNamespace(name=name or None, arguments=args or None)
    tc = SimpleNamespace(index=idx, id=call_id or None, function=fn)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]))]
    )


class ScriptedProvider(LLMProvider):
    """Replays a list of streams, one per `stream_chat_completion` call."""

    def __init__(self, scripts: list[list[SimpleNamespace]]) -> None:
        self._scripts = list(scripts)
        self.call_count = 0

    def chat_completion(self, capability: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        raise AssertionError("non-streaming path should not be called in this test")

    def stream_chat_completion(
        self,
        capability: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[SimpleNamespace]:
        self.call_count += 1
        return iter(self._scripts.pop(0))

    @property
    def supports_vision(self) -> bool:
        return False

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def client(self) -> Any:
        return None


def test_loop_yields_text_when_no_tools():
    provider = ScriptedProvider([[_text_chunk("hello "), _text_chunk("world")]])
    loop = ToolLoop(llm_provider=provider, tool_registry=ToolRegistry(), tools_enabled=False)

    messages = [{"role": "user", "content": "hi"}]
    chunks = list(loop.stream(messages))

    assert chunks == ["hello ", "world"]
    assert messages[-1] == {"role": "assistant", "content": "hello world"}
    assert provider.call_count == 1


def test_loop_runs_tool_call_then_final_reply():
    registry = ToolRegistry()
    registry.register(
        name="get_time",
        description="Returns the time",
        parameters={"type": "object", "properties": {}},
        handler=lambda: "2026-05-09T17:11",
    )

    first_stream = [
        _tool_chunk(0, call_id="call_abc", name="get_time", args="{}"),
    ]
    second_stream = [
        _text_chunk("the time is "),
        _text_chunk("2026-05-09"),
    ]

    provider = ScriptedProvider([first_stream, second_stream])
    loop = ToolLoop(llm_provider=provider, tool_registry=registry, tools_enabled=True)

    messages: list[dict[str, Any]] = [{"role": "user", "content": "what time is it"}]
    chunks = list(loop.stream(messages))

    assert chunks == ["the time is ", "2026-05-09"]
    assert provider.call_count == 2

    # Conversation history should include: user, assistant(tool_calls), tool, assistant(reply)
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "get_time"
    assert messages[2]["content"] == "2026-05-09T17:11"
    assert messages[3]["content"] == "the time is 2026-05-09"


def test_loop_complete_collects_streamed_text():
    provider = ScriptedProvider([[_text_chunk("foo"), _text_chunk("bar")]])
    loop = ToolLoop(llm_provider=provider, tool_registry=ToolRegistry(), tools_enabled=False)
    assert loop.complete([{"role": "user", "content": "x"}]) == "foobar"


def test_loop_falls_back_when_streaming_unsupported():
    class NoStreamProvider(LLMProvider):
        def chat_completion(self, capability, messages, **kwargs):
            message = SimpleNamespace(
                content="non-streamed reply",
                tool_calls=None,
            )
            message.model_dump = lambda: {"role": "assistant", "content": "non-streamed reply"}
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        def stream_chat_completion(self, capability, messages, **kwargs):
            raise NotImplementedError

        @property
        def supports_vision(self):
            return False

        @property
        def supports_tools(self):
            return True

        @property
        def client(self):
            return None

    loop = ToolLoop(llm_provider=NoStreamProvider(), tool_registry=ToolRegistry(), tools_enabled=False)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    chunks = list(loop.stream(messages))
    assert "".join(chunks) == "non-streamed reply"
    assert messages[-1]["content"] == "non-streamed reply"


def test_loop_assembles_split_tool_call_chunks():
    """Tool call deltas often arrive split across many chunks — verify
    that the accumulator stitches name/arguments back together."""
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        handler=lambda text: text.upper(),
    )

    first_stream = [
        _tool_chunk(0, call_id="c1", name="ec"),
        _tool_chunk(0, name="ho", args='{"tex'),
        _tool_chunk(0, args='t":"hi"}'),
    ]
    second_stream = [_text_chunk("done")]

    provider = ScriptedProvider([first_stream, second_stream])
    loop = ToolLoop(llm_provider=provider, tool_registry=registry, tools_enabled=True)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "say hi"}]

    list(loop.stream(messages))

    tool_message = next(m for m in messages if m["role"] == "tool")
    assert tool_message["content"] == "HI"
    assert tool_message["tool_call_id"] == "c1"
