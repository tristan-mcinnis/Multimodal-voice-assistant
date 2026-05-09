# Domain Glossary

The vocabulary used in this codebase. Use these terms exactly when naming
modules, classes, or methods so that the intent stays legible in conversation,
diffs, and ADRs.

## Pipeline concepts

- **Wake word**: the spoken trigger ("nova" by default) that gates whether a
  transcribed phrase becomes a prompt.
- **Prompt**: the user-facing text that follows the wake word, before any
  enrichment with conversation history or tool output.
- **Turn**: one round of (assistant request → optional tool calls → assistant
  reply). A single user prompt can produce multiple turns when the model
  decides to call tools.
- **Voice command**: a recognised prompt that bypasses the LLM (e.g.,
  `remember …`, `forget context`, `search …`). Currently inlined in
  `VoiceAssistant._handle_command`.

## LLM seam

- **LLM provider**: an adapter that satisfies the `LLMProvider` interface
  (`chat_completion`, `stream_chat_completion`, capability flags).
- **OpenAI-compatible endpoint**: any service speaking the OpenAI Chat
  Completions wire format. DeepSeek, LM Studio, and OpenAI itself all qualify
  and share the `OpenAICompatibleProvider` adapter.
- **Capability**: the *kind* of work a model is good at — `conversation`,
  `vision`, `structured`. Each provider keeps an ordered preference list per
  capability in its `ModelCatalog`.
- **Model catalog**: the per-provider preference list of model IDs. The first
  model that succeeds wins; failing models are skipped with a log entry.
- **Tool loop**: the module that owns the LLM↔tools dance — `ToolLoop.stream`
  yields assistant text chunks and folds tool-call results back into the
  conversation until the model returns a clean reply.

## Tool seam

- **Tool registry**: ordered registration of named tools, each with a JSON
  schema and a Python handler. The registry produces OpenAI-formatted tool
  descriptors and dispatches calls by name.
- **Tool**: a named function exposed to the LLM via tool-calling. Built-ins
  cover clipboard, web search, screenshot, and webcam capture.

## TTS seam

- **TTS provider**: adapter satisfying the `TTSProvider` interface (`speak`,
  `stream_speak`). Kokoro and OpenAI implementations exist.
- **Streaming TTS**: synthesis that consumes a chunked text iterator and
  produces audio with sentence-level latency, rather than waiting for the full
  response.
