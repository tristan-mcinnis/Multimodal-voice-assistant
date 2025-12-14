"""
Multi-modal AI voice assistant package.

Supports both cloud (OpenAI) and fully local (LM Studio + Kokoro) operation.
"""

__version__ = "0.2.0"

from .core import VoiceAssistant, main

__all__ = [
    "__version__",
    "VoiceAssistant",
    "main",
]
