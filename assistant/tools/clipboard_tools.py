"""Clipboard tool implementations."""

from __future__ import annotations

from typing import Optional

import pyperclip

from ..utils import log


def get_clipboard_text() -> Optional[str]:
    """Extract text from the system clipboard.

    Returns:
        The clipboard text, or None if empty or unavailable
    """
    log("Extracting clipboard text...", title="ACTION", style="bold blue")
    try:
        clipboard_content = pyperclip.paste()
    except pyperclip.PyperclipException as exc:
        log(f"Clipboard access failed: {exc}", title="ERROR", style="bold red")
        return None
    if isinstance(clipboard_content, str) and clipboard_content.strip():
        log("Clipboard text extracted.", title="ACTION", style="bold blue")
        return clipboard_content
    log("No clipboard text to copy", title="ERROR", style="bold red")
    return None


def extract_clipboard_text_tool() -> str:
    """Tool handler for clipboard text extraction.

    Returns:
        The clipboard text or an error message
    """
    text = get_clipboard_text()
    if text:
        return text
    return "Clipboard is empty or unavailable."
