"""Abstract base class for TTS providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TTSProvider(ABC):
    """Abstract base class for text-to-speech providers."""

    @abstractmethod
    def speak(self, text: str) -> bool:
        """Synthesize and play speech.

        Args:
            text: The text to convert to speech

        Returns:
            True if speech was successfully synthesized and played, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this TTS provider."""
        pass
