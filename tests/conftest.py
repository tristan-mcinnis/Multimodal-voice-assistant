"""Shared pytest configuration.

The voice assistant relies on env-var-driven config that's evaluated at
import time. Set safe defaults BEFORE any `assistant.*` module loads so
tests can run without a real API key or LM Studio process.
"""

from __future__ import annotations

import os

# Pick a provider that won't try to phone home or auto-detect a local model
# during import. `deepseek` only constructs a client when the provider is
# actually instantiated (not at module import).
os.environ.setdefault("LLM_PROVIDER", "deepseek")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("ASSISTANT_TTS_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
