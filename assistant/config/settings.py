"""Configuration settings loaded from environment variables.

Provider-specific knowledge (model catalogs) lives next to the provider that
uses it; this module is just env-var parsing.
"""

from __future__ import annotations

import os
from typing import Dict, List


# -----------------------------------------------------------------------------
# Wake word & system prompt
# -----------------------------------------------------------------------------
WAKE_WORD = os.getenv("ASSISTANT_WAKE_WORD", "nova").strip().lower()

SYSTEM_MESSAGE = (
    "You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context "
    "(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed "
    "text prompt that will be attached to their transcribed voice prompt. Generate the most useful and "
    "factual response possible, carefully considering all previous generated text in your response before "
    "adding new tokens to the response. Do not expect or request images, just use the context if added. "
    "Use all of the context of this conversation so your response is relevant to the conversation. Make "
    "your responses clear and concise, avoiding any verbosity."
)

NUM_CORES = max(os.cpu_count() or 2, 2)


# -----------------------------------------------------------------------------
# LLM Provider Configuration
# -----------------------------------------------------------------------------
# DeepSeek is the default — fast, cheap, OpenAI-compatible.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_PREFERRED_CHAT_MODEL = os.getenv("DEEPSEEK_PREFERRED_CHAT_MODEL", "deepseek-v4-flash")
DEEPSEEK_PREFERRED_REASONING_MODEL = os.getenv("DEEPSEEK_PREFERRED_REASONING_MODEL", "deepseek-v4-pro")

# Local LLM (LM Studio)
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "local-model")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_PREFERRED_CHAT_MODEL = os.getenv("CLAUDE_PREFERRED_CHAT_MODEL", "claude-opus-4-5-20251101")
CLAUDE_PREFERRED_VISION_MODEL = os.getenv("CLAUDE_PREFERRED_VISION_MODEL", "claude-opus-4-5-20251101")
CLAUDE_PREFERRED_TOOL_MODEL = os.getenv("CLAUDE_PREFERRED_TOOL_MODEL", "claude-sonnet-4-20250514")


# -----------------------------------------------------------------------------
# TTS Provider Configuration
# -----------------------------------------------------------------------------
TTS_PROVIDER = os.getenv("ASSISTANT_TTS_PROVIDER", "kokoro").strip().lower()

KOKORO_CLI_PATH = os.getenv("KOKORO_CLI_PATH", "kokoro-tts").strip() or "kokoro-tts"
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")
KOKORO_LANGUAGE = os.getenv("KOKORO_LANGUAGE", "en-us")
KOKORO_SPEED = os.getenv("KOKORO_SPEED", "1.0")
KOKORO_MODEL_PATH = (os.getenv("KOKORO_MODEL_PATH") or "").strip()
KOKORO_VOICES_PATH = (os.getenv("KOKORO_VOICES_PATH") or "").strip()

KOKORO_STREAMING = os.getenv("KOKORO_STREAMING", "false").strip().lower() in {"1", "true", "yes"}
KOKORO_ONNX_MODEL_PATH = os.getenv("KOKORO_ONNX_MODEL_PATH", "").strip()
KOKORO_VOICES_BIN_PATH = os.getenv("KOKORO_VOICES_BIN_PATH", "").strip()


# -----------------------------------------------------------------------------
# Tool Configuration
# -----------------------------------------------------------------------------
ENABLE_TOOL_CALLING = os.getenv("ASSISTANT_DISABLE_TOOLS", "0").lower() not in {"1", "true", "yes"}
SIMPLE_TOOLS = os.getenv("ASSISTANT_SIMPLE_TOOLS", "false").strip().lower() in {"1", "true", "yes"}


# -----------------------------------------------------------------------------
# MCP / Context Provider Configuration
# -----------------------------------------------------------------------------
MCP_CONTEXT_FILE = os.getenv("MCP_CONTEXT_FILE", "").strip()
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "").strip()
MCP_DEFAULT_NAMESPACE = os.getenv("MCP_DEFAULT_NAMESPACE", "").strip()


# -----------------------------------------------------------------------------
# Model Catalog
# -----------------------------------------------------------------------------
class ModelCatalog:
    """Ordered preference list per capability ('conversation', 'vision', 'structured')."""

    def __init__(self, catalog: Dict[str, List[str]]) -> None:
        self._catalog = {capability: self._dedupe(models) for capability, models in catalog.items()}

    @staticmethod
    def _dedupe(models: List[str]) -> List[str]:
        seen: set = set()
        ordered: List[str] = []
        for model in models:
            if not model or model in seen:
                continue
            ordered.append(model)
            seen.add(model)
        return ordered

    def get(self, capability: str) -> List[str]:
        return self._catalog.get(capability, [])


# Provider-owned catalogs. Each provider imports its own; consumers should not
# care which provider is active.

DEEPSEEK_MODEL_CATALOG = ModelCatalog({
    "conversation": [DEEPSEEK_PREFERRED_CHAT_MODEL, DEEPSEEK_PREFERRED_REASONING_MODEL],
    "vision": [],  # DeepSeek does not currently expose vision via this endpoint.
    "structured": [DEEPSEEK_PREFERRED_CHAT_MODEL, DEEPSEEK_PREFERRED_REASONING_MODEL],
})

OPENAI_MODEL_CATALOG = ModelCatalog({
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

ANTHROPIC_MODEL_CATALOG = ModelCatalog({
    "conversation": [CLAUDE_PREFERRED_CHAT_MODEL, "claude-sonnet-4-20250514"],
    "vision": [CLAUDE_PREFERRED_VISION_MODEL, "claude-sonnet-4-20250514"],
    "structured": [CLAUDE_PREFERRED_TOOL_MODEL, "claude-sonnet-4-20250514"],
})

# Default catalog is whichever provider is active. Kept for backwards
# compatibility with code that still imports DEFAULT_MODEL_CATALOG; new code
# should reach for the provider-specific catalog directly.
if LLM_PROVIDER == "anthropic":
    DEFAULT_MODEL_CATALOG = ANTHROPIC_MODEL_CATALOG
elif LLM_PROVIDER == "openai":
    DEFAULT_MODEL_CATALOG = OPENAI_MODEL_CATALOG
elif LLM_PROVIDER == "local":
    DEFAULT_MODEL_CATALOG = ModelCatalog({
        "conversation": [LOCAL_LLM_MODEL],
        "vision": [],
        "structured": [LOCAL_LLM_MODEL],
    })
else:  # deepseek (default) and unknown providers
    DEFAULT_MODEL_CATALOG = DEEPSEEK_MODEL_CATALOG
