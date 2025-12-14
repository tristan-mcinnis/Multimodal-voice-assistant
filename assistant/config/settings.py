"""Configuration settings loaded from environment variables."""

from __future__ import annotations

import os
from typing import Dict, List


# Wake word configuration
WAKE_WORD = os.getenv("ASSISTANT_WAKE_WORD", "nova").strip().lower()

# System message for GPT models
SYSTEM_MESSAGE = (
    "You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context "
    "(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed "
    "text prompt that will be attached to their transcribed voice prompt. Generate the most useful and "
    "factual response possible, carefully considering all previous generated text in your response before "
    "adding new tokens to the response. Do not expect or request images, just use the context if added. "
    "Use all of the context of this conversation so your response is relevant to the conversation. Make "
    "your responses clear and concise, avoiding any verbosity."
)

# CPU cores for Whisper
NUM_CORES = max(os.cpu_count() or 2, 2)

# -----------------------------------------------------------------------------
# LLM Provider Configuration
# -----------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").strip().lower()

# Local LLM settings (LM Studio)
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "local-model")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Anthropic settings (for future Claude integration)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_PREFERRED_CHAT_MODEL = os.getenv("CLAUDE_PREFERRED_CHAT_MODEL", "claude-opus-4-5-20251101")
CLAUDE_PREFERRED_VISION_MODEL = os.getenv("CLAUDE_PREFERRED_VISION_MODEL", "claude-opus-4-5-20251101")
CLAUDE_PREFERRED_TOOL_MODEL = os.getenv("CLAUDE_PREFERRED_TOOL_MODEL", "claude-sonnet-4-20250514")

# -----------------------------------------------------------------------------
# TTS Provider Configuration
# -----------------------------------------------------------------------------
TTS_PROVIDER = os.getenv("ASSISTANT_TTS_PROVIDER", "openai").strip().lower()

# Kokoro TTS settings
KOKORO_CLI_PATH = os.getenv("KOKORO_CLI_PATH", "kokoro-tts").strip() or "kokoro-tts"
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")
KOKORO_LANGUAGE = os.getenv("KOKORO_LANGUAGE", "en-us")
KOKORO_SPEED = os.getenv("KOKORO_SPEED", "1.0")
KOKORO_MODEL_PATH = (os.getenv("KOKORO_MODEL_PATH") or "").strip()
KOKORO_VOICES_PATH = (os.getenv("KOKORO_VOICES_PATH") or "").strip()

# Streaming Kokoro TTS configuration (uses kokoro-onnx for low latency)
KOKORO_STREAMING = os.getenv("KOKORO_STREAMING", "false").strip().lower() in {"1", "true", "yes"}
KOKORO_ONNX_MODEL_PATH = os.getenv("KOKORO_ONNX_MODEL_PATH", "").strip()
KOKORO_VOICES_BIN_PATH = os.getenv("KOKORO_VOICES_BIN_PATH", "").strip()

# -----------------------------------------------------------------------------
# Tool Configuration
# -----------------------------------------------------------------------------
ENABLE_TOOL_CALLING = os.getenv("ASSISTANT_DISABLE_TOOLS", "0").lower() not in {"1", "true", "yes"}
SIMPLE_TOOLS = os.getenv("ASSISTANT_SIMPLE_TOOLS", "false").strip().lower() in {"1", "true", "yes"}

# -----------------------------------------------------------------------------
# Context Provider Configuration (MCP)
# -----------------------------------------------------------------------------
MCP_CONTEXT_FILE = os.getenv("MCP_CONTEXT_FILE", "").strip()
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "").strip()
MCP_DEFAULT_NAMESPACE = os.getenv("MCP_DEFAULT_NAMESPACE", "").strip()


# -----------------------------------------------------------------------------
# Model Catalog
# -----------------------------------------------------------------------------
class ModelCatalog:
    """Maintain ordered preference lists for different model capabilities."""

    def __init__(self, catalog: Dict[str, List[str]]) -> None:
        self._catalog = {capability: self._dedupe(models) for capability, models in catalog.items()}

    @staticmethod
    def _dedupe(models: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for model in models:
            if not model:
                continue
            if model not in seen:
                ordered.append(model)
                seen.add(model)
        return ordered

    def get(self, capability: str) -> List[str]:
        return self._catalog.get(capability, [])


# Default model catalogs per provider
if LLM_PROVIDER == "local":
    DEFAULT_MODEL_CATALOG = ModelCatalog({
        "conversation": [LOCAL_LLM_MODEL],
        "vision": [],  # Vision not typically supported by local models
        "structured": [LOCAL_LLM_MODEL],
    })
elif LLM_PROVIDER == "anthropic":
    DEFAULT_MODEL_CATALOG = ModelCatalog({
        "conversation": [CLAUDE_PREFERRED_CHAT_MODEL, "claude-sonnet-4-20250514"],
        "vision": [CLAUDE_PREFERRED_VISION_MODEL, "claude-sonnet-4-20250514"],
        "structured": [CLAUDE_PREFERRED_TOOL_MODEL, "claude-sonnet-4-20250514"],
    })
else:
    DEFAULT_MODEL_CATALOG = ModelCatalog({
        "conversation": [
            os.getenv("OPENAI_PREFERRED_CHAT_MODEL", "gpt-5"),
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4o",
        ],
        "vision": [
            os.getenv("OPENAI_PREFERRED_VISION_MODEL", "gpt-5"),
            "gpt-5-mini",
            "gpt-4o",
        ],
        "structured": [
            os.getenv("OPENAI_PREFERRED_TOOL_MODEL", "gpt-5-mini"),
            "gpt-5-nano",
            "gpt-4o-mini",
            "gpt-4o",
        ],
    })
