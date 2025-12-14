"""Media capture utilities (screenshot, webcam)."""

from .screenshot import take_screenshot, encode_image
from .webcam import web_cam_capture

__all__ = [
    "take_screenshot",
    "encode_image",
    "web_cam_capture",
]
