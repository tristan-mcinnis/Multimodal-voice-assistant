# ADR 0001 — Consolidate OpenAI-compatible LLM providers behind one adapter

- **Status**: Accepted (2026-05-09)
- **Deciders**: tristan-mcinnis

## Context

Adding DeepSeek as an LLM provider would have created a third file
(`deepseek_provider.py`) duplicating ~90% of the logic in
`openai_provider.py` and `local_provider.py`: same `OpenAI` SDK client,
same fallback loop, same params shape.

While auditing the existing code we also noticed `OpenAIProvider` did not
override `stream_chat_completion`, so cloud OpenAI silently fell back to the
non-streaming path — breaking the streaming TTS pipeline that
`LocalProvider` already supported.

## Decision

Introduce `OpenAICompatibleProvider`, a single adapter that takes
`(name, api_key, base_url, catalog, http_client?, supports_vision?,
supports_tools?)` and satisfies the `LLMProvider` interface — both
streaming and non-streaming.

The concrete providers become thin factories:

- `OpenAIProvider` — no `base_url`, OpenAI's own catalog.
- `LocalProvider` — adds model auto-detection and a localhost-bypass
  `httpx` client.
- `DeepSeekProvider` — sets `base_url=https://api.deepseek.com` and the
  DeepSeek catalog (default `deepseek-v4-flash`).

## Consequences

- One place owns the OpenAI Chat Completions wire format. Streaming, model
  fallback, and request-shape decisions live behind a single seam.
- Adding a new OpenAI-compatible service is a config edit + a
  ~15-line factory file, not a new copy of the loop.
- Cloud OpenAI now streams by default, fixing the latent TTS-pipelining bug.
- Provider-specific knowledge (model catalogs) moved out of
  `config/settings.py` into per-provider catalogs, so settings is now pure
  env-var parsing.

## Alternatives considered

- **Subclass + abstract method**: rejected — encourages future drift between
  providers as each one overrides "just one more thing." A flat config object
  passed at construction time is harder to slip provider-specific logic into.
- **Skip the refactor and just add a 4th clone for DeepSeek**: rejected — the
  duplicate-fix problem (streaming bug only in one of three near-identical
  files) is a strong signal the modules are too shallow.
