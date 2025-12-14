"""Kokoro TTS provider implementation (CLI and ONNX streaming)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Optional

import pyaudio

from .base import TTSProvider
from ...config import (
    KOKORO_CLI_PATH,
    KOKORO_VOICE,
    KOKORO_LANGUAGE,
    KOKORO_SPEED,
    KOKORO_MODEL_PATH,
    KOKORO_VOICES_PATH,
    KOKORO_STREAMING,
    KOKORO_ONNX_MODEL_PATH,
    KOKORO_VOICES_BIN_PATH,
)
from ...utils import log


# Lazy-loaded Kokoro ONNX instance for streaming TTS
_kokoro_onnx_instance: Optional[Any] = None


def _find_model_file(filename: str) -> str:
    """Find a model file, checking models/ directory first."""
    # Check models/ directory first
    models_path = Path("models") / filename
    if models_path.exists():
        return str(models_path)
    # Fall back to current directory
    if Path(filename).exists():
        return filename
    # Return models/ path as default (will show better error message)
    return str(models_path)


def _get_kokoro_onnx() -> Optional[Any]:
    """Lazy initialization of Kokoro ONNX for streaming TTS."""
    global _kokoro_onnx_instance
    if _kokoro_onnx_instance is None:
        try:
            from kokoro_onnx import Kokoro
            model_path = KOKORO_ONNX_MODEL_PATH or _find_model_file("kokoro-v1.0.onnx")
            voices_path = KOKORO_VOICES_BIN_PATH or _find_model_file("voices-v1.0.bin")
            _kokoro_onnx_instance = Kokoro(model_path, voices_path)
            log("Initialized Kokoro ONNX streaming TTS", title="TTS", style="bold blue")
        except Exception as exc:
            log(f"Failed to initialize Kokoro ONNX: {exc}", title="TTS", style="bold red")
            return None
    return _kokoro_onnx_instance


def _play_wav_file(audio_path: Path) -> None:
    """Play a WAV file using PyAudio."""
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


class KokoroProvider(TTSProvider):
    """Kokoro TTS provider with CLI and ONNX streaming support."""

    def __init__(self) -> None:
        self._use_streaming = KOKORO_STREAMING

    @property
    def name(self) -> str:
        return "kokoro"

    def speak(self, text: str) -> bool:
        """Speak text using Kokoro TTS with fallback to OpenAI."""
        # Try streaming mode first for lower latency
        if self._use_streaming:
            if self._speak_streaming(text):
                return True
            log("Streaming Kokoro failed. Trying CLI fallback.", title="TTS", style="bold yellow")

        # Fall back to CLI-based Kokoro
        if self._speak_cli(text):
            return True

        return False

    def _speak_streaming(self, text: str) -> bool:
        """Stream audio using kokoro-onnx for low-latency playback."""
        kokoro = _get_kokoro_onnx()
        if kokoro is None:
            return False

        try:
            import sounddevice as sd

            voice = (KOKORO_VOICE or "af_sarah").strip() or "af_sarah"
            speed_value = (KOKORO_SPEED or "1.0").strip() or "1.0"
            try:
                speed = float(speed_value)
            except ValueError:
                speed = 1.0

            # Split text into sentences for pseudo-streaming (low latency on first output)
            sentences = re.split(r'(?<=[.!?])\s+', text)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                try:
                    samples, sample_rate = kokoro.create(
                        sentence,
                        voice=voice,
                        speed=speed,
                        lang=KOKORO_LANGUAGE or "en-us",
                    )
                    sd.play(samples, sample_rate)
                    sd.wait()
                except Exception as exc:
                    log(f"Kokoro streaming sentence failed: {exc}", title="TTS", style="bold yellow")
                    continue

            return True

        except ImportError:
            log("sounddevice not installed. Run: pip install sounddevice", title="TTS", style="bold red")
            return False
        except Exception as exc:
            log(f"Kokoro streaming TTS failed: {exc}", title="TTS", style="bold red")
            return False

    def _speak_cli(self, text: str) -> bool:
        """Speak text using the Kokoro CLI."""
        kokoro_executable = shutil.which(KOKORO_CLI_PATH)
        if not kokoro_executable:
            log(
                f"Kokoro CLI '{KOKORO_CLI_PATH}' not found. Install kokoro-tts and ensure it is on your PATH.",
                title="TTS",
                style="bold yellow",
            )
            return False

        voice = (KOKORO_VOICE or "af_sarah").strip() or "af_sarah"
        language = (KOKORO_LANGUAGE or "en-us").strip() or "en-us"
        speed_value = (KOKORO_SPEED or "1.0").strip() or "1.0"
        try:
            float(speed_value)
        except ValueError:
            log(
                f"Invalid KOKORO_SPEED '{speed_value}'. Falling back to 1.0.",
                title="TTS",
                style="bold yellow",
            )
            speed_value = "1.0"

        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        audio_path = Path(temp_audio_path)

        command = [
            kokoro_executable,
            "-",
            str(audio_path),
            "--format",
            "wav",
            "--lang",
            language,
            "--voice",
            voice,
            "--speed",
            speed_value,
        ]

        model_path = KOKORO_MODEL_PATH
        if model_path:
            command.extend(["--model", model_path])

        voices_path = KOKORO_VOICES_PATH
        if voices_path:
            command.extend(["--voices", voices_path])

        log(
            f"Generating speech with Kokoro CLI voice '{voice}' ({language}).",
            title="TTS",
            style="bold blue",
        )

        try:
            result = subprocess.run(
                command,
                input=text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 - surface CLI errors gracefully
            log(f"Failed to run Kokoro CLI: {exc}", title="TTS", style="bold red")
            audio_path.unlink(missing_ok=True)
            return False

        if result.returncode != 0:
            error_output = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            log(
                f"Kokoro CLI exited with status {result.returncode}: {error_output}",
                title="TTS",
                style="bold red",
            )
            audio_path.unlink(missing_ok=True)
            return False

        if not audio_path.exists():
            log("Kokoro CLI did not produce an audio file.", title="TTS", style="bold red")
            return False

        try:
            _play_wav_file(audio_path)
        except Exception as exc:  # noqa: BLE001 - playback errors should be surfaced
            log(f"Failed to play Kokoro audio output: {exc}", title="TTS", style="bold red")
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False

        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass
        return True
