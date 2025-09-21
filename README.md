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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
