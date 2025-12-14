"""Tool registry for assistant function calling."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from ..utils import log


class ToolRegistry:
    """Registry for assistant tools exposed to OpenAI tool calling."""

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]],
        handler: Callable[..., str],
    ) -> None:
        """Register a new tool.

        Args:
            name: Unique name for the tool
            description: Description of what the tool does
            parameters: JSON Schema for the tool parameters
            handler: Function to handle tool execution
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters if parameters is not None else {"type": "object", "properties": {}},
            "handler": handler,
        }

    def has_tools(self) -> bool:
        """Check if any tools are registered."""
        return bool(self._tools)

    def as_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert registered tools to OpenAI tool format."""
        tools = []
        for tool in self._tools.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
            )
        return tools

    def execute(self, name: str, arguments_json: str) -> str:
        """Execute a tool by name.

        Args:
            name: The tool name to execute
            arguments_json: JSON string of arguments

        Returns:
            The tool result as a string
        """
        if name not in self._tools:
            log(f"Tool '{name}' is not registered.", title="TOOL", style="bold red")
            return f"Tool '{name}' is not available."

        handler = self._tools[name]["handler"]
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            if not isinstance(args, dict):
                args = {"value": args}
        except json.JSONDecodeError:
            args = {}

        log(f"Running tool '{name}' with args {args}.", title="TOOL", style="bold cyan")
        try:
            result = handler(**args)
        except Exception as exc:  # noqa: BLE001 - surface tool execution errors to the model
            log(f"Tool '{name}' failed: {exc}", title="TOOL", style="bold red")
            return f"Tool '{name}' execution failed: {exc}"

        if result is None:
            return ""
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
