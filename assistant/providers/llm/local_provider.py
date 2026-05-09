"""Local LLM provider (LM Studio compatible) — wraps the OpenAI-compatible adapter.

Adds two local-specific behaviours on top of the shared adapter:
  - Auto-detects the loaded model via the `models.list` endpoint when no
    `LOCAL_LLM_MODEL` is set explicitly.
  - Forces an httpx client that bypasses system proxies for localhost.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx

from .openai_compatible import OpenAICompatibleProvider
from ...config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_MODEL,
    ModelCatalog,
)
from ...utils import log


def _build_localhost_http_client() -> httpx.Client:
    """An httpx client that ignores HTTP_PROXY for localhost."""
    return httpx.Client(
        mounts={
            "http://localhost": httpx.HTTPTransport(),
            "http://127.0.0.1": httpx.HTTPTransport(),
        }
    )


class LocalProvider(OpenAICompatibleProvider):
    """LM-Studio-style local LLM."""

    def __init__(self) -> None:
        super().__init__(
            name="local",
            api_key=LOCAL_LLM_API_KEY,
            base_url=LOCAL_LLM_BASE_URL,
            catalog=ModelCatalog({"conversation": [LOCAL_LLM_MODEL], "vision": [], "structured": [LOCAL_LLM_MODEL]}),
            http_client=_build_localhost_http_client(),
            supports_vision=False,
            supports_tools=True,
        )
        log(f"Using local LLM at {LOCAL_LLM_BASE_URL}", title="LLM", style="bold blue")

        detected = self._detect_model()
        if detected:
            self._catalog = ModelCatalog({
                "conversation": [detected],
                "vision": [],
                "structured": [detected],
            })
            log(f"Auto-detected model: {detected}", title="LLM", style="bold blue")

    def _detect_model(self) -> Optional[str]:
        """Pick the first non-embedding model exposed by the local endpoint."""
        if LOCAL_LLM_MODEL != "local-model":
            return LOCAL_LLM_MODEL  # explicit override — trust it.

        for attempt in range(3):
            try:
                models = self._client.models.list()
                for model in models.data:
                    model_id = model.id.lower()
                    if "embed" not in model_id and "ocr" not in model_id:
                        return model.id
                if models.data:
                    return models.data[0].id
            except Exception as exc:  # noqa: BLE001 - detection is best-effort
                if attempt < 2:
                    time.sleep(1)
                else:
                    log(
                        f"Model detection failed after 3 attempts: {exc}",
                        title="LLM",
                        style="bold yellow",
                    )
        return None
