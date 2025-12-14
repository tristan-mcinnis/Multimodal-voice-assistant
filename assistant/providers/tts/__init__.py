"""TTS provider factory and abstractions."""

from __future__ import annotations

from .base import TTSProvider
from ...config import TTS_PROVIDER


def get_tts_provider() -> TTSProvider:
    """Factory function to get the configured TTS provider."""
    if TTS_PROVIDER == "kokoro":
        from .kokoro_tts import KokoroProvider
        return KokoroProvider()
    else:
        from .openai_tts import OpenAITTSProvider
        return OpenAITTSProvider()


__all__ = [
    "TTSProvider",
    "get_tts_provider",
]
