"""Speech recognition and audio utilities."""

from .recognition import wav_to_text, extract_prompt, get_whisper_model
from .audio import play_wav_file

__all__ = [
    "wav_to_text",
    "extract_prompt",
    "get_whisper_model",
    "play_wav_file",
]
