"""Tool registry round-trip tests."""

from __future__ import annotations

from assistant.tools.registry import ToolRegistry


def test_registry_starts_empty():
    registry = ToolRegistry()
    assert not registry.has_tools()
    assert registry.as_openai_tools() == []


def test_registry_registers_and_executes_tool():
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Echo input",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda text: f"echoed: {text}",
    )

    assert registry.has_tools()
    tools = registry.as_openai_tools()
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "echo"

    result = registry.execute("echo", '{"text": "hi"}')
    assert result == "echoed: hi"


def test_registry_handles_missing_tool():
    registry = ToolRegistry()
    result = registry.execute("nope", "{}")
    assert "not available" in result


def test_registry_serializes_dict_results():
    registry = ToolRegistry()
    registry.register(
        name="data",
        description="returns dict",
        parameters={"type": "object", "properties": {}},
        handler=lambda: {"a": 1},
    )
    assert registry.execute("data", "{}") == '{"a": 1}'


def test_registry_surfaces_handler_errors():
    registry = ToolRegistry()

    def boom():
        raise ValueError("kaboom")

    registry.register(
        name="boom",
        description="raises",
        parameters={"type": "object", "properties": {}},
        handler=boom,
    )
    result = registry.execute("boom", "{}")
    assert "execution failed" in result
    assert "kaboom" in result
