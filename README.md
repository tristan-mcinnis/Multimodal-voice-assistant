# Multi-Modal AI Voice Assistant

A multi-modal AI voice assistant supporting multiple LLM providers (OpenAI, local LM Studio, Claude/Anthropic) with configurable text-to-speech (OpenAI streaming or Kokoro). Combines voice transcription, tool calling, clipboard extraction, screenshot analysis, and web search to respond with rich context.

## Features

- **Multi-provider LLM support**: OpenAI (GPT-5), local models via LM Studio, Claude/Anthropic
- **Tool calling**: Screenshot capture, webcam capture, clipboard extraction, DuckDuckGo search
- **Flexible TTS**: OpenAI streaming voices or offline Kokoro synthesis
- **Model Context Protocol (MCP)**: Pluggable context providers for external integrations
- **Wake word activation**: Say "nova" followed by your prompt
- **Graceful fallbacks**: Models and TTS providers fall back automatically on failure

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

### Cloud mode (OpenAI)

```bash
export OPENAI_API_KEY="sk-..."
export LLM_PROVIDER=openai
export ASSISTANT_TTS_PROVIDER=openai
python run.py
```

### Local mode (LM Studio + Kokoro)

```bash
# 1. Start LM Studio and load a model
# 2. Download Kokoro TTS models (one-time, ~335MB)
curl -L -o kokoro-v1.0.onnx https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

# 3. Run the assistant
python run.py
```

The wake word is **"nova"**. Say it followed by your request.

## Configuration

### LLM Providers

```bash
# OpenAI (default)
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
| `LLM_PROVIDER` | `openai`, `local`, or `anthropic` |
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
├── core.py               # VoiceAssistant orchestrator
├── config/settings.py    # Environment configuration
├── providers/
│   ├── llm/              # OpenAI, local, Anthropic providers
│   └── tts/              # OpenAI, Kokoro providers
├── tools/                # Tool registry and implementations
├── context/              # Conversation and MCP context
├── speech/               # Whisper recognition
├── media/                # Screenshot, webcam capture
└── utils/                # Logging, message helpers
```

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

For offline text-to-speech, you need to download the model files (~335MB total):

```bash
# Download model files to the project directory
curl -L -o kokoro-v1.0.onnx https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
```

That's it! Kokoro is the default TTS provider and will automatically find these files in the project directory.

### Optional: Custom model location

If you prefer to store the models elsewhere:

```bash
# Download to a custom location
mkdir -p ~/kokoro
curl -L -o ~/kokoro/kokoro-v1.0.onnx https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
curl -L -o ~/kokoro/voices-v1.0.bin https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

# Set environment variables to point to your files
export KOKORO_ONNX_MODEL_PATH=~/kokoro/kokoro-v1.0.onnx
export KOKORO_VOICES_BIN_PATH=~/kokoro/voices-v1.0.bin
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
