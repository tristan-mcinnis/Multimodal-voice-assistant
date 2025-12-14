"""Screenshot capture utilities."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from PIL import ImageGrab

from ..utils import log


def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def take_screenshot(output_path: str = "screenshot.jpg") -> Optional[str]:
    """Capture a screenshot and save it to the specified path.

    Args:
        output_path: Path where the screenshot will be saved

    Returns:
        The path to the saved screenshot, or None if capture failed
    """
    log("Taking screenshot...", title="ACTION", style="bold blue")
    try:
        screenshot = ImageGrab.grab()
        rgb_screenshot = screenshot.convert("RGB")
        rgb_screenshot.save(output_path, quality=15)
        return output_path
    except Exception as exc:  # noqa: BLE001 - handle headless environments gracefully
        log(f"Error taking screenshot: {exc}", title="ERROR", style="bold red")
        return None
