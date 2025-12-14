"""Tool registry and implementations for the assistant."""

from .registry import ToolRegistry
from .vision_tools import capture_screenshot_context_tool, capture_webcam_context_tool, vision_prompt
from .search_tools import duckduckgo_search, duckduckgo_search_tool, process_search_results
from .clipboard_tools import get_clipboard_text, extract_clipboard_text_tool

__all__ = [
    "ToolRegistry",
    "capture_screenshot_context_tool",
    "capture_webcam_context_tool",
    "vision_prompt",
    "duckduckgo_search",
    "duckduckgo_search_tool",
    "process_search_results",
    "get_clipboard_text",
    "extract_clipboard_text_tool",
]
