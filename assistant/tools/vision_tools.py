"""Vision-related tool implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..media import take_screenshot, web_cam_capture, encode_image
from ..utils import log, message_to_dict, extract_message_text

if TYPE_CHECKING:
    from ..providers.llm import LLMProvider


# Module-level provider reference (set during initialization)
_llm_provider: "LLMProvider" = None


def set_llm_provider(provider: "LLMProvider") -> None:
    """Set the LLM provider for vision operations."""
    global _llm_provider
    _llm_provider = provider


def vision_prompt(prompt: str, photo_path: str) -> str:
    """Generate a description of an image using the vision model.

    Args:
        prompt: The context prompt to guide the image analysis
        photo_path: Path to the image file

    Returns:
        A description of the image relevant to the prompt
    """
    if _llm_provider is None:
        log("LLM provider not initialized for vision.", title="ERROR", style="bold red")
        return ""

    if not _llm_provider.supports_vision:
        log("Current LLM provider does not support vision.", title="ERROR", style="bold yellow")
        return ""

    log("Generating vision prompt...", title="ACTION", style="bold blue")
    try:
        encoded_image = encode_image(photo_path)
    except FileNotFoundError:
        log("Image file not found for vision prompt.", title="ERROR", style="bold red")
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image in the context of the following prompt: "
                        f"{prompt}. Provide a detailed description focusing on elements relevant to the prompt."
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encoded_image}", "detail": "high"}},
            ],
        }
    ]
    try:
        response = _llm_provider.chat_completion(
            "vision",
            messages=messages,
            max_tokens=300,
        )
    except Exception as exc:  # noqa: BLE001 - surface the failure but keep running
        log(f"Vision model failed: {exc}", title="ERROR", style="bold red")
        return ""
    message_dict = message_to_dict(response.choices[0].message)
    description = extract_message_text(message_dict)
    log("Vision prompt generated.", title="ACTION", style="bold blue")
    return description


def capture_screenshot_context_tool(user_prompt: str = "") -> str:
    """Capture a screenshot and describe it for context.

    Args:
        user_prompt: The user's prompt to guide the screenshot analysis

    Returns:
        Description of the screenshot for the assistant
    """
    photo_path = take_screenshot()
    if not photo_path:
        return "Screenshot capture failed."
    description = vision_prompt(user_prompt or "Describe this screenshot for the assistant.", photo_path)
    if not description:
        description = "No description available."
    return f"Screenshot stored at {photo_path}. Description: {description}"


def capture_webcam_context_tool(user_prompt: str = "") -> str:
    """Capture a webcam image and describe it for context.

    Args:
        user_prompt: The user's prompt to guide the webcam analysis

    Returns:
        Description of the webcam image for the assistant
    """
    photo_path = web_cam_capture()
    if not photo_path:
        return "Webcam capture failed."
    description = vision_prompt(user_prompt or "Describe this webcam image for the assistant.", photo_path)
    if not description:
        description = "No description available."
    return f"Webcam image stored at {photo_path}. Description: {description}"
