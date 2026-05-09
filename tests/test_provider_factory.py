"""Smoke tests for the LLM provider factory and DeepSeek configuration."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def reload_assistant_modules():
    """Reload assistant.config + assistant.providers.llm so tests can flip env vars.

    Because the modules read env vars at import time, importing them once
    captures whatever was in os.environ at that moment. This fixture forces
    a clean reload after the test mutates env vars.
    """

    def _reload():
        for mod_name in [
            "assistant.providers.llm",
            "assistant.providers.llm.deepseek_provider",
            "assistant.providers.llm.openai_compatible",
            "assistant.config",
            "assistant.config.settings",
        ]:
            if mod_name in importlib.sys.modules:
                importlib.reload(importlib.sys.modules[mod_name])

    yield _reload
    _reload()


def test_deepseek_is_default_provider(monkeypatch, reload_assistant_modules):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    reload_assistant_modules()

    from assistant.config import DEEPSEEK_PREFERRED_CHAT_MODEL, LLM_PROVIDER

    assert LLM_PROVIDER == "deepseek"
    assert DEEPSEEK_PREFERRED_CHAT_MODEL == "deepseek-v4-flash"


def test_deepseek_provider_uses_deepseek_base_url(monkeypatch, reload_assistant_modules):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    reload_assistant_modules()

    from assistant.providers.llm import get_llm_provider

    provider = get_llm_provider()
    assert provider.name == "deepseek"
    assert str(provider.client.base_url).startswith("https://api.deepseek.com")
    assert "deepseek-v4-flash" in provider.catalog.get("conversation")


def test_factory_rejects_unknown_provider(monkeypatch, reload_assistant_modules):
    monkeypatch.setenv("LLM_PROVIDER", "totally-fake")
    reload_assistant_modules()

    from assistant.providers.llm import get_llm_provider

    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        get_llm_provider()


def test_deepseek_requires_api_key(monkeypatch, reload_assistant_modules):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    reload_assistant_modules()

    from assistant.providers.llm import get_llm_provider

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        get_llm_provider()
