# Multi-Modal AI Voice Assistant

A multi-modal AI voice assistant supporting **DeepSeek (default)**, OpenAI, Anthropic Claude, and local LM Studio LLMs with configurable text-to-speech (OpenAI streaming or Kokoro). Combines voice transcription, tool calling, clipboard extraction, screenshot analysis, and web search to respond with rich context.

## Features

- **Multi-provider LLM support**: DeepSeek (default, fast & cheap), OpenAI (GPT-5), local LM Studio, Anthropic Claude
- **Tool calling**: Screenshot capture, webcam capture, clipboard extraction, DuckDuckGo search
- **Flexible TTS**: OpenAI streaming voices or offline Kokoro synthesis
- **Model Context Protocol (MCP)**: Pluggable context providers for external integrations
- **Wake word activation**: Say "nova" followed by your prompt
- **Graceful fallbacks**: Models and TTS providers fall back automatically on failure
- **`.env` support**: All credentials/config can live in a single gitignored `.env` file

## Installation

```bash
# Clone the repository
git clone https://github.com/tristan-mcinnis/Multimodal-voice-assistant
cd Multimodal-voice-assistant

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

### Default mode (DeepSeek + Kokoro)

```bash
# 1. Copy the example .env and add your DeepSeek key
cp .env.example .env
# Edit .env, set DEEPSEEK_API_KEY=sk-...

# 2. (Optional) Download Kokoro TTS models for offline speech (~335MB)
mkdir -p models
curl -L -o models/kokoro-v1.0.onnx https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

# 3. Run the assistant
python run.py
```

Get a DeepSeek API key at <https://platform.deepseek.com/>. The default model is
`deepseek-v4-flash` (1M context, low-latency). Override with
`DEEPSEEK_PREFERRED_CHAT_MODEL=deepseek-v4-pro` for the higher-capability model.

### Cloud mode (OpenAI)

```bash
export OPENAI_API_KEY="sk-..."
export LLM_PROVIDER=openai
python run.py
```

### Local mode (LM Studio)

```bash
# Start LM Studio and load a model first.
export LLM_PROVIDER=local
export LOCAL_LLM_BASE_URL=http://localhost:1234/v1
python run.py
```

The wake word is **"nova"**. Say it followed by your request.

## Configuration

The assistant reads a `.env` file in the project root (loaded via
`python-dotenv` before any submodule imports). Anything you can `export` you
can also drop in `.env`. See [`.env.example`](./.env.example) for the full
template — `.env` itself is gitignored.

### LLM Providers

```bash
# DeepSeek (default — set in .env)
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY="sk-..."
export DEEPSEEK_PREFERRED_CHAT_MODEL=deepseek-v4-flash  # or deepseek-v4-pro

# OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

# Local LM Studio
export LLM_PROVIDER=local
export LOCAL_LLM_BASE_URL=http://localhost:1234/v1
export LOCAL_LLM_MODEL=your-model-name

# Claude/Anthropic
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Text-to-Speech

```bash
# Kokoro TTS (default, local, offline)
# Requires downloading model files - see "Kokoro TTS Setup" below
export ASSISTANT_TTS_PROVIDER=kokoro
export KOKORO_VOICE=af_sarah
export KOKORO_STREAMING=true  # Low-latency ONNX mode

# OpenAI TTS (requires API key)
export ASSISTANT_TTS_PROVIDER=openai
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `deepseek` (default), `openai`, `local`, or `anthropic` |
| `DEEPSEEK_API_KEY` | DeepSeek API key (default provider) |
| `DEEPSEEK_PREFERRED_CHAT_MODEL` | Default `deepseek-v4-flash` |
| `DEEPSEEK_BASE_URL` | Override (default `https://api.deepseek.com`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude) |
| `LOCAL_LLM_BASE_URL` | LM Studio endpoint (default: `http://localhost:1234/v1`) |
| `LOCAL_LLM_MODEL` | Model name in LM Studio |
| `ASSISTANT_TTS_PROVIDER` | `openai` or `kokoro` |
| `ASSISTANT_DISABLE_TOOLS` | Set to `true` to disable tool calling |
| `ASSISTANT_SIMPLE_TOOLS` | Set to `true` for clipboard + search only (no vision) |
| `MCP_CONTEXT_FILE` | Path to MCP context file |

## Architecture

```
assistant/
├── core.py                                # VoiceAssistant orchestrator
├── config/settings.py                     # Env-var parsing
├── providers/
│   ├── llm/
│   │   ├── openai_compatible.py           # Shared adapter for OpenAI-shaped APIs
│   │   ├── deepseek_provider.py           # DeepSeek (default)
│   │   ├── openai_provider.py             # OpenAI
│   │   ├── local_provider.py              # LM Studio
│   │   └── anthropic_provider.py          # Claude
│   └── tts/                               # OpenAI, Kokoro
├── tools/
│   ├── loop.py                            # Unified streaming tool-call loop
│   ├── registry.py                        # Tool registry
│   └── …                                  # clipboard, search, vision tools
├── context/                               # Conversation and MCP context
├── speech/                                # Whisper recognition
├── media/                                 # Screenshot, webcam capture
└── utils/                                 # Logging, message helpers
```

DeepSeek, OpenAI, and LM Studio all speak the OpenAI Chat Completions wire
format and share `OpenAICompatibleProvider` — see
[ADR 0001](./docs/adr/0001-consolidate-openai-compatible-providers.md). The
streaming + non-streaming tool-call loop lives behind one seam — see
[ADR 0002](./docs/adr/0002-unified-tool-loop.md). The domain glossary is in
[CONTEXT.md](./CONTEXT.md).

## Testing

```bash
pip install -e '.[dev]'
pytest
```

Tests cover the tool registry, the unified `ToolLoop` (with a scripted fake
provider — no network/audio needed), and the LLM provider factory.

## Extending

### Adding Tools

Register tools in `VoiceAssistant._register_builtin_tools()` or create new files in `assistant/tools/`:

```python
self.tool_registry.register(
    name="my_tool",
    description="What this tool does",
    parameters={"type": "object", "properties": {...}},
    handler=lambda **kwargs: "result",
)
```

### Adding LLM Providers

1. Create `assistant/providers/llm/my_provider.py`
2. Implement the `LLMProvider` interface from `base.py`
3. Update the factory in `assistant/providers/llm/__init__.py`

## Kokoro TTS Setup

For offline text-to-speech, download the model files (~335MB total) to the `models/` directory:

```bash
mkdir -p models
curl -L -o models/kokoro-v1.0.onnx https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
```

That's it! Kokoro is the default TTS provider and will automatically find these files.

### Optional: Custom model location

If you prefer to store the models elsewhere:

```bash
export KOKORO_ONNX_MODEL_PATH=/path/to/kokoro-v1.0.onnx
export KOKORO_VOICES_BIN_PATH=/path/to/voices-v1.0.bin
```

### Streaming mode (lower latency)

For reduced latency, enable streaming mode:

```bash
export KOKORO_STREAMING=true
```

## Dependencies

Core: `openai`, `faster-whisper`, `SpeechRecognition`, `pyaudio`, `rich`, `Pillow`, `pygame`, `duckduckgo-search`, `scikit-learn`

Optional: `kokoro-tts`, `kokoro-onnx`, `anthropic`

## Credits

- [Kokoro TTS](https://github.com/nazdridoy/kokoro-tts) by Nazmus Sakib Dridoy

## License

MIT License - see [LICENSE](LICENSE) for details.
