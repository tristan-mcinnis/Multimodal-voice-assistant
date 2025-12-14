"""OpenAI TTS provider implementation."""

from __future__ import annotations

import pyaudio
from openai import OpenAI

from .base import TTSProvider
from ...config import OPENAI_API_KEY
from ...utils import log


class OpenAITTSProvider(TTSProvider):
    """OpenAI streaming text-to-speech provider."""

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI TTS")
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    @property
    def name(self) -> str:
        return "openai"

    def speak(self, text: str) -> bool:
        """Stream audio from OpenAI TTS API."""
        player = pyaudio.PyAudio()
        stream = None
        try:
            stream = player.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
            stream_started = False
            try:
                with self._client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="nova",
                    response_format="pcm",
                    speed="1.75",
                    input=text,
                ) as response:
                    silence_threshold = 0.01
                    for chunk in response.iter_bytes(chunk_size=1024):
                        if stream_started:
                            stream.write(chunk)
                        else:
                            if chunk and max(chunk) > silence_threshold:
                                stream.write(chunk)
                                stream_started = True
            except Exception as exc:  # noqa: BLE001 - surface TTS errors gracefully
                log(f"OpenAI TTS failed: {exc}", title="TTS", style="bold red")
                return False
            return True
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            player.terminate()
