# Multi-Modal AI Voice Assistant

This project is a multi-modal AI voice assistant that uses OpenAI's GPT-4o, audio processing with WhisperModel, speech recognition, clipboard extraction, and image processing to respond to user prompts.

## Installation

To install the required dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/nexuslux/Multimodal-voice-assistant
    cd multimodal-voice-assistant
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. Obtain an OpenAI API key and replace the placeholder in the code:
    ```python
    openai_client = OpenAI(api_key="sk-your-api-key")
    ```

## Usage

1. Run the script:
    ```bash
    python your_script_name.py
    ```

2. The assistant will listen for the wake word (`nova`) followed by your prompt. It supports various functionalities such as taking screenshots, capturing webcam images, and extracting clipboard content.

## Features

- **Wake Word Detection:** Starts listening for commands when the wake word 'nova' is detected.
- **Screenshot Capture:** Takes a screenshot and processes it for context.
- **Webcam Capture:** Captures an image from the webcam and processes it for context.
- **Clipboard Extraction:** Extracts text from the clipboard for additional context.
- **Enhanced Conversation Context:** Maintains a summary of previous exchanges for coherent responses.

## Dependencies

- `openai`: OpenAI API client
- `Pillow`: Image processing
- `faster-whisper`: Whisper model for audio transcription
- `SpeechRecognition`: Speech recognition
- `pyperclip`: Clipboard handling
- `opencv-python-headless`: Computer vision
- `pyaudio`: Audio handling
- `rich`: Rich text formatting in the terminal
- `pygame`: Image capture from webcam
- `duckduckgo-search`: DuckDuckGo search integration
- `scikit-learn`: Machine learning utilities
- `numpy`: Numerical computations

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
