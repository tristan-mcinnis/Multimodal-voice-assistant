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

    def stream_speak(self, text_stream: any) -> bool:
        """Synthesize and play speech from a text stream (iterator).

        Args:
            text_stream: Iterator yielding text chunks

        Returns:
            True if successful, False otherwise
        """
        # Default fallback: accumulate text and speak
        full_text = "".join(text_stream)
        if full_text:
            return self.speak(full_text)
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this TTS provider."""
        pass
