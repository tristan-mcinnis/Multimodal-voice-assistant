"""Webcam capture utilities."""

from __future__ import annotations

from typing import Optional

from PIL import Image

from ..utils import log


def web_cam_capture(output_path: str = "webcam.jpg") -> Optional[str]:
    """Capture an image from the webcam.

    Args:
        output_path: Path where the webcam image will be saved

    Returns:
        The path to the saved image, or None if capture failed
    """
    try:
        import pygame
        import pygame.camera

        pygame.camera.init()
        cameras = pygame.camera.list_cameras()

        if not cameras:
            log("Error: No cameras found", title="ERROR", style="bold red")
            return None

        cam = pygame.camera.Camera(cameras[0], (640, 480))
        cam.start()
        image = cam.get_image()
        cam.stop()
        pygame.camera.quit()

        pygame.image.save(image, output_path)

        pil_string_image = pygame.image.tostring(image, "RGB", False)
        pil_image = Image.frombytes("RGB", (640, 480), pil_string_image)

        pil_image.save(output_path, "JPEG")

        log("Webcam image captured and saved.", title="ACTION", style="bold blue")
        return output_path
    except ImportError:
        log("pygame not installed. Run: pip install pygame", title="ERROR", style="bold red")
        return None
    except Exception as exc:  # noqa: BLE001 - webcam is optional
        log(f"Error capturing webcam image: {exc}", title="ERROR", style="bold red")
        return None
