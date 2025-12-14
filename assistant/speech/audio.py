"""Audio playback utilities."""

from __future__ import annotations

import wave
from pathlib import Path

import pyaudio


def play_wav_file(audio_path: Path) -> None:
    """Play a WAV file using PyAudio.

    Args:
        audio_path: Path to the WAV file to play
    """
    player = pyaudio.PyAudio()
    stream = None
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            stream = player.open(
                format=player.get_format_from_width(wav_file.getsampwidth()),
                channels=wav_file.getnchannels(),
                rate=wav_file.getframerate(),
                output=True,
            )
            data = wav_file.readframes(1024)
            while data:
                stream.write(data)
                data = wav_file.readframes(1024)
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        player.terminate()
