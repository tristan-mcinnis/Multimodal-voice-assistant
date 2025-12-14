"""Message normalization utilities for different API response formats."""

from __future__ import annotations

from typing import Any, Dict, List


def message_to_dict(message: Any) -> Dict[str, Any]:
    """Convert a message object to a dictionary."""
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if hasattr(message, "to_dict"):
        return message.to_dict()
    raise TypeError("Unsupported message type for serialization.")


def extract_message_text(message: Dict[str, Any]) -> str:
    """Extract text content from a message dictionary."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    text_parts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            else:
                text_parts.append(str(part))
    return "".join(text_parts).strip()


def iter_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and normalize tool calls from a message."""
    tool_calls = message.get("tool_calls") or []
    normalized: List[Dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            normalized.append(call)
        elif hasattr(call, "model_dump"):
            normalized.append(call.model_dump())
        elif hasattr(call, "to_dict"):
            normalized.append(call.to_dict())
    return normalized
