"""
Multi-modal AI voice assistant package.

Supports DeepSeek (default), OpenAI, Anthropic Claude, and local LM Studio
models with configurable TTS (OpenAI streaming or Kokoro).
"""

# Load .env BEFORE any submodule imports so env-var-driven config picks it up.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional — env vars exported by the shell still work.
    pass

__version__ = "0.3.0"

from .core import VoiceAssistant, main

__all__ = [
    "__version__",
    "VoiceAssistant",
    "main",
]
