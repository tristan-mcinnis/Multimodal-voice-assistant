"""DeepSeek LLM provider (OpenAI-compatible endpoint).

Docs: https://api-docs.deepseek.com/

Default model is `deepseek-v4-flash` (latest, low-latency, 1M context).
`deepseek-v4-pro` is the higher-capability alternative.
"""

from __future__ import annotations

import os

from ...config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL_CATALOG,
)
from .openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider — OpenAI-compatible chat completions."""

    def __init__(self) -> None:
        # Re-read at construction time so tests can flip the env var without
        # reloading the whole package. python-dotenv will have populated
        # os.environ from .env before this runs.
        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable is required when LLM_PROVIDER=deepseek"
            )
        super().__init__(
            name="deepseek",
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            catalog=DEEPSEEK_MODEL_CATALOG,
            supports_vision=False,
            supports_tools=True,
        )
