# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-modal AI voice assistant that supports cloud (OpenAI), local (LM Studio + Kokoro), and Claude (Anthropic) operation. Uses Whisper for local speech-to-text and configurable TTS (OpenAI streaming or Kokoro). Listens for the wake word "nova", then processes voice commands with support for clipboard extraction, web search, and optional screenshot/webcam analysis via tool calling.

## Running the Assistant

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .

# Run with entry point script
python run.py

# Or run as a module
python -m assistant

# Cloud mode (requires OPENAI_API_KEY)
export LLM_PROVIDER=openai
python -m assistant

# Local mode (LM Studio + Kokoro for low latency)
export LLM_PROVIDER=local
export LOCAL_LLM_BASE_URL=http://localhost:1234/v1
export LOCAL_LLM_MODEL=your-model-name
export ASSISTANT_TTS_PROVIDER=kokoro
export ASSISTANT_SIMPLE_TOOLS=true
python -m assistant

# Claude mode (requires ANTHROPIC_API_KEY)
export LLM_PROVIDER=anthropic
python -m assistant
```

## Architecture

**Modular package architecture** in the `assistant/` directory.

### Directory Structure

```
assistant/
├── __init__.py           # Package init, exports VoiceAssistant, main
├── __main__.py           # Entry point: python -m assistant
├── core.py               # Main VoiceAssistant class and orchestration
├── config/
│   └── settings.py       # All env vars, ModelCatalog, defaults
├── providers/
│   ├── llm/
│   │   ├── base.py           # Abstract LLMProvider class
│   │   ├── openai_provider.py
│   │   ├── local_provider.py
│   │   └── anthropic_provider.py
│   └── tts/
│       ├── base.py           # Abstract TTSProvider class
│       ├── openai_tts.py
│       └── kokoro_tts.py
├── tools/
│   ├── registry.py       # ToolRegistry class
│   ├── vision_tools.py   # Screenshot, webcam tools
│   ├── search_tools.py   # DuckDuckGo search
│   └── clipboard_tools.py
├── context/
│   ├── conversation.py   # EnhancedConversationContext
│   └── mcp.py            # ContextProvider, MCPContextProvider
├── speech/
│   ├── recognition.py    # Whisper transcription
│   └── audio.py          # Audio playback
├── media/
│   ├── screenshot.py     # PIL screenshot capture
│   └── webcam.py         # Pygame webcam capture
└── utils/
    ├── logging.py        # Rich logging
    └── messages.py       # Message normalization
```

### Key Components

- **`assistant/core.py`**: `VoiceAssistant` class orchestrates the entire pipeline. Contains `complete_chat_with_tools()` for the tool calling loop and `llm_prompt()` for context assembly.

- **`assistant/providers/llm/`**: Pluggable LLM providers with automatic fallback. Factory function `get_llm_provider()` returns the configured provider.

- **`assistant/providers/tts/`**: Pluggable TTS providers. `KokoroProvider` supports both CLI and ONNX streaming modes.

- **`assistant/tools/registry.py`**: `ToolRegistry` class for OpenAI function calling. Tools are registered in `VoiceAssistant._register_builtin_tools()`.

- **`assistant/context/`**: Conversation context with TF-IDF topic detection and MCP integration.

### Adding New Tools

Register tools in `VoiceAssistant._register_builtin_tools()` (in `core.py`) or create a new tool file in `assistant/tools/`:

```python
self.tool_registry.register(
    name="tool_name",
    description="What the tool does",
    parameters={"type": "object", "properties": {...}},
    handler=lambda **kwargs: "result string",
)
```

### Adding a New LLM Provider

1. Create `assistant/providers/llm/your_provider.py`
2. Implement the `LLMProvider` abstract class from `base.py`
3. Update `assistant/providers/llm/__init__.py` factory function

## Environment Variables

### LLM Configuration
| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `openai`, `local`, or `anthropic` |
| `OPENAI_API_KEY` | Required when LLM_PROVIDER=openai |
| `ANTHROPIC_API_KEY` | Required when LLM_PROVIDER=anthropic |
| `LOCAL_LLM_BASE_URL` | LM Studio endpoint (default: http://localhost:1234/v1) |
| `LOCAL_LLM_MODEL` | Model name loaded in LM Studio |
| `OPENAI_PREFERRED_CHAT_MODEL` | Override primary chat model (default: gpt-5) |
| `CLAUDE_PREFERRED_CHAT_MODEL` | Override Claude chat model (default: claude-opus-4-5) |

### TTS Configuration
| Variable | Purpose |
|----------|---------|
| `ASSISTANT_TTS_PROVIDER` | `openai` (default) or `kokoro` |
| `KOKORO_VOICE` | Voice name (default: af_sarah) |
| `KOKORO_SPEED` | Speech speed (default: 1.0) |
| `KOKORO_STREAMING` | Set to `true` for low-latency ONNX streaming |
| `KOKORO_ONNX_MODEL_PATH` | Path to kokoro ONNX model file |
| `KOKORO_VOICES_BIN_PATH` | Path to kokoro voices directory |

### Tools Configuration
| Variable | Purpose |
|----------|---------|
| `ASSISTANT_DISABLE_TOOLS` | Set to `1`/`true`/`yes` to disable tool calling |
| `ASSISTANT_SIMPLE_TOOLS` | Set to `true` to only enable clipboard and web search |
| `MCP_CONTEXT_FILE` | Path to file for MCP context injection |

## Key Patterns

- **Provider abstraction**: LLM and TTS providers implement abstract interfaces for easy swapping
- **Graceful degradation**: Models fall back to alternatives on failure; TTS falls back from Kokoro to OpenAI
- **Factory functions**: `get_llm_provider()` and `get_tts_provider()` instantiate configured providers
- **Tool execution loop**: `complete_chat_with_tools()` handles the OpenAI tool calling loop
- **Voice pipeline**: Microphone → Whisper → wake word extraction → LLM → TTS

## Logs

Logs are saved to `logs/YYYY/MM/DD-HHMMSS.log` with a `latest.log` symlink.
