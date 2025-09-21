# Multi-Modal AI Voice Assistant

This project is a multi-modal AI voice assistant that now targets OpenAI's latest GPT-5 family of models while maintaining
backwards-compatible fallbacks. The assistant combines voice transcription, tool calling, clipboard extraction, screenshot
analysis, and search to respond to user prompts with rich context.

## What's new

- **GPT-5 ready:** Automatically prefers `gpt-5`, `gpt-5-mini`, and `gpt-5-nano` with graceful fallbacks to earlier models when
  needed.
- **Tool calling architecture:** Tools such as screenshot capture, webcam capture, clipboard extraction, and DuckDuckGo search
  are exposed through OpenAI's function/tool calling and are easy to extend.
- **Model Context Protocol (MCP) hooks:** A pluggable context provider allows you to surface MCP context (via files or custom
  clients) alongside the conversation history.
- **Environment-driven configuration:** Select models, toggle tool usage, or provide MCP metadata without modifying the source
  code.
- **Offline Kokoro text-to-speech:** Optionally route responses through the open-source Kokoro CLI for high-quality on-device
  synthesis.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/nexuslux/Multimodal-voice-assistant
    cd multimodal-voice-assistant
    ```

2. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Set the required and optional environment variables before launching the assistant. On macOS you can export these variables in
your shell profile (e.g. `~/.zshrc`).

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key (no code changes required). |
| `OPENAI_PREFERRED_CHAT_MODEL` | ❌ | Override the primary chat model (defaults to `gpt-5`). |
| `OPENAI_PREFERRED_VISION_MODEL` | ❌ | Override the model used for vision analysis. |
| `OPENAI_PREFERRED_TOOL_MODEL` | ❌ | Override the model used when the assistant performs structured tool reasoning. |
| `ASSISTANT_DISABLE_TOOLS` | ❌ | Set to `1`, `true`, or `yes` to disable tool calling entirely. |
| `ASSISTANT_TTS_PROVIDER` | ❌ | Choose `openai` (default) or `kokoro` for text-to-speech output. |
| `KOKORO_CLI_PATH` | ❌ | Path to the `kokoro-tts` executable if it is not already on your `PATH`. |
| `KOKORO_VOICE` / `KOKORO_LANGUAGE` / `KOKORO_SPEED` | ❌ | Configure the Kokoro voice (default `af_sarah`), language (`en-us`), and speed (`1.0`). |
| `KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH` | ❌ | Point to downloaded Kokoro ONNX and voices files when using the CLI provider. |
| `MCP_CONTEXT_FILE` | ❌ | Path to a file whose contents should be injected as MCP context. |
| `MCP_ENDPOINT` / `MCP_DEFAULT_NAMESPACE` | ❌ | Placeholders for integrating a live MCP server or service. |

Example (macOS / Linux):
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_PREFERRED_CHAT_MODEL="gpt-5"
```

## Usage

Run the assistant from the project root:
```bash
python assistant.py
```
The wake word is `nova`. After saying the wake word, speak your request. The assistant can:

- Record audio and transcribe it locally with `faster-whisper`.
- Capture macOS screenshots and analyse them with a GPT-5 vision model.
- Capture webcam images (if available) and analyse them.
- Read clipboard contents to enrich responses.
- Perform DuckDuckGo web searches via tool calling.

### Text-to-speech options

The assistant streams speech with OpenAI's `tts-1` voice by default. To run everything locally you can switch to the
[`kokoro-tts`](https://github.com/nazdridoy/kokoro-tts) CLI, which is now integrated as an alternate output path.

1. Install the Kokoro CLI (already listed in `requirements.txt`, but you can also install it manually):
   ```bash
   pip install kokoro-tts
   ```
2. Download the Kokoro model and voice assets. On macOS you can keep them in `~/Library/Application Support/kokoro`:
   ```bash
   mkdir -p "$HOME/Library/Application Support/kokoro"
   cd "$HOME/Library/Application Support/kokoro"
   curl -L -O https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
   curl -L -O https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
   ```
3. Export the environment variables so the assistant can find and configure the CLI:
   ```bash
   export ASSISTANT_TTS_PROVIDER=kokoro
   export KOKORO_MODEL_PATH="$HOME/Library/Application Support/kokoro/kokoro-v1.0.onnx"
   export KOKORO_VOICES_PATH="$HOME/Library/Application Support/kokoro/voices-v1.0.bin"
   export KOKORO_VOICE=af_sarah   # pick any supported Kokoro voice
   export KOKORO_LANGUAGE=en-us    # optional, defaults to en-us
   export KOKORO_SPEED=1.0         # optional, defaults to 1.0
   ```

If the Kokoro CLI cannot be executed the assistant automatically falls back to OpenAI's streaming voice so you are never left
without audio output.

### Extending tools

Tools are registered in `register_builtin_tools()` inside `assistant.py`. Each tool maps metadata to a Python handler. To add a
new tool, register it with the `ToolRegistry` and return a string (or JSON-serialisable object) describing the result. The GPT-5
models will automatically call registered tools when appropriate.

### Model Context Protocol (MCP)

The `MCPContextProvider` class offers an extension point for integrating the Model Context Protocol. By default it can read a
context file specified via `MCP_CONTEXT_FILE`. Advanced MCP clients can be implemented by extending `MCPContextProvider` to
retrieve context from remote MCP services (referenced via `MCP_ENDPOINT` / `MCP_DEFAULT_NAMESPACE`). Any collected MCP context is
merged with conversation history before prompting GPT-5.

## Features

- **Wake word detection** via `SpeechRecognition`.
- **Adaptive conversation memory** with similarity-based reset.
- **Mac-first screenshot capture** using Pillow's `ImageGrab` (works on macOS and Windows with a display).
- **Webcam capture** leveraging `pygame`.
- **Tool calling & automation** using OpenAI's tool interface.
- **Flexible text-to-speech** with OpenAI's streaming voices or the Kokoro CLI running locally.
- **MCP-ready context providers** for future integrations.

## Dependencies

- `openai>=1.40.0`
- `Pillow`
- `faster-whisper`
- `SpeechRecognition`
- `pyperclip`
- `opencv-python-headless`
- `pyaudio`
- `rich`
- `pygame`
- `duckduckgo-search`
- `scikit-learn`
- `numpy`
- `kokoro-tts`

## Credits

- [Kokoro TTS CLI](https://github.com/nazdridoy/kokoro-tts) by [Nazmus Sakib Dridoy](https://github.com/nazdridoy) powers the optional local speech synthesis path exposed by this assistant.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
