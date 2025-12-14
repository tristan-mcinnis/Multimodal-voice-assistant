# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-modal AI voice assistant that supports both cloud (OpenAI) and fully local (LM Studio + Kokoro) operation. Uses Whisper for local speech-to-text and configurable TTS (OpenAI streaming or Kokoro). Listens for the wake word "nova", then processes voice commands with support for clipboard extraction, web search, and optional screenshot/webcam analysis via tool calling.

## Running the Assistant

```bash
# Install dependencies
pip install -r requirements.txt

# Cloud mode (requires OPENAI_API_KEY)
python assistant.py

# Local mode (LM Studio + Kokoro for low latency)
export LLM_PROVIDER=local
export LOCAL_LLM_BASE_URL=http://localhost:1234/v1
export LOCAL_LLM_MODEL=your-model-name
export ASSISTANT_TTS_PROVIDER=kokoro
export ASSISTANT_SIMPLE_TOOLS=true
python assistant.py
```

## Architecture

**Single-file architecture** - all code lives in `assistant.py` (~900 lines).

### Key Components

- **ModelCatalog / OpenAIModelManager** (lines 156-204): Model preference lists with automatic fallback between models (gpt-5 → gpt-5-mini → gpt-5-nano → gpt-4o).

- **ToolRegistry** (lines 229-293): Central registry for OpenAI function calling. Tools are registered with `register_builtin_tools()` at line 557. Each tool has metadata and a handler function.

- **ContextProvider / MCPContextProvider** (lines 298-359): Pluggable context injection system for Model Context Protocol integration. Currently supports reading from a file (`MCP_CONTEXT_FILE` env var).

- **EnhancedConversationContext** (lines 91-150): Manages conversation history with TF-IDF similarity-based topic change detection and automatic context clearing.

- **TTS System** (lines 696-841): Dual-provider text-to-speech with OpenAI streaming (`speak_with_openai`) and Kokoro CLI (`speak_with_kokoro`) with automatic fallback.

### Adding New Tools

Register tools in `register_builtin_tools()` using:

```python
tool_registry.register(
    name="tool_name",
    description="What the tool does",
    parameters={"type": "object", "properties": {...}},
    handler=lambda **kwargs: "result string",
)
```

The GPT models will automatically call tools when appropriate. Tool handlers should return strings or JSON-serializable objects.

## Environment Variables

### LLM Configuration
| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `openai` (default) or `local` for LM Studio |
| `OPENAI_API_KEY` | Required when LLM_PROVIDER=openai |
| `LOCAL_LLM_BASE_URL` | LM Studio endpoint (default: http://localhost:1234/v1) |
| `LOCAL_LLM_MODEL` | Model name loaded in LM Studio |
| `OPENAI_PREFERRED_CHAT_MODEL` | Override primary chat model (default: gpt-5) |
| `OPENAI_PREFERRED_VISION_MODEL` | Override vision model |
| `OPENAI_PREFERRED_TOOL_MODEL` | Override structured/tool model |

### TTS Configuration
| Variable | Purpose |
|----------|---------|
| `ASSISTANT_TTS_PROVIDER` | `openai` (default) or `kokoro` |
| `KOKORO_VOICE` | Voice name (default: af_sarah). Options: af_nicole, am_adam, am_michael, bf_emma, bm_george |
| `KOKORO_SPEED` | Speech speed (default: 1.0) |
| `KOKORO_STREAMING` | Set to `true` for low-latency streaming mode (requires kokoro-onnx) |
| `KOKORO_ONNX_MODEL_PATH` | Path to kokoro ONNX model file |
| `KOKORO_VOICES_BIN_PATH` | Path to kokoro voices directory |

### Tools Configuration
| Variable | Purpose |
|----------|---------|
| `ASSISTANT_DISABLE_TOOLS` | Set to `1`/`true`/`yes` to disable tool calling |
| `ASSISTANT_SIMPLE_TOOLS` | Set to `true` to only enable clipboard and web search (skip vision tools) |
| `MCP_CONTEXT_FILE` | Path to file for MCP context injection |

## Key Patterns

- **Graceful degradation**: Models fall back to alternatives on failure; TTS falls back from Kokoro to OpenAI.
- **Tool execution loop**: `complete_chat_with_tools()` (line 620) handles the OpenAI tool calling loop, executing tools and feeding results back until no more tool calls are requested.
- **Voice pipeline**: Microphone → Whisper transcription → wake word extraction → GPT response → TTS output.
