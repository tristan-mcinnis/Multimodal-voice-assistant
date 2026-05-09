# ADR 0002 — Unified `ToolLoop` for streaming and non-streaming tool calls

- **Status**: Accepted (2026-05-09)
- **Deciders**: tristan-mcinnis

## Context

`VoiceAssistant.complete_chat_with_tools` and
`VoiceAssistant.stream_chat_with_tools` re-implemented the same loop:
build params, send to the LLM, append tool messages, loop until the model
stops calling tools. The streaming version had a hand-rolled chunk
accumulator (~20 lines) that drifted from the non-streaming code path.

A bug fix in one path silently failed to reach the other.

## Decision

Extract `assistant/tools/loop.py::ToolLoop`. It owns the entire LLM↔tools
exchange behind one method (`stream`) that yields assistant text chunks
live and mutates the message list with the full conversation. The
non-streaming entry point (`complete`) is `"".join(stream(...))`.

If a provider raises `NotImplementedError` from `stream_chat_completion`,
the loop transparently falls back to `chat_completion` and yields the
final text in one chunk.

## Consequences

- One place owns the tool-loop semantics. Tests can drive it with a fake
  provider and a fake registry — no microphone, no Whisper, no audio
  hardware.
- `VoiceAssistant` shrinks: the streaming/non-streaming methods are gone,
  and `llm_prompt` becomes a 12-line orchestration of (build-prompt →
  loop.stream → tts.stream_speak).
- The streaming chunk accumulator now lives in one place — bug fixes
  apply to all callers automatically.
