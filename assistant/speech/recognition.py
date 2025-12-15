"""Speech recognition using Whisper."""

from __future__ import annotations

import re
from typing import Any, Optional, Union, BinaryIO

import numpy as np

from ..config import NUM_CORES


# Lazy-loaded Whisper model
_whisper_model: Optional[Any] = None


def get_whisper_model() -> Any:
    """Get or initialize the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            cpu_threads=NUM_CORES // 2,
            num_workers=NUM_CORES // 2,
        )
    return _whisper_model


def wav_to_text(audio_input: Union[str, BinaryIO, np.ndarray]) -> str:
    """Transcribe audio to text using Whisper.

    Args:
        audio_input: Path to the WAV file, a file-like object, or numpy array to transcribe

    Returns:
        The transcribed text
    """
    whisper_model = get_whisper_model()
    segments, _ = whisper_model.transcribe(audio_input)
    text = "".join(segment.text for segment in segments)
    return text


def extract_prompt(transcribed_text: str, wake_word: str) -> Optional[str]:
    """Extract the user's prompt from transcribed text after the wake word.

    Args:
        transcribed_text: The full transcribed text
        wake_word: The wake word to look for

    Returns:
        The prompt text after the wake word, or None if wake word not found
    """
    pattern = rf"\b{re.escape(wake_word)}[\s,.?!]*(.*)"
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    if match:
        prompt = match.group(1).strip()
        return prompt
    return None
