from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pyperclip
import pyaudio
import speech_recognition as sr
from duckduckgo_search import DDGS
from faster_whisper import WhisperModel
from openai import OpenAI
from PIL import Image, ImageGrab
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Rich Console
console = Console()

# Store logs in memory
log_messages: List[str] = []


def log(message: str, title: str, style: str) -> None:
    """Log a message to the Rich console and persist it for later export."""
    console.print(Panel(Markdown(f"**{message}**"), border_style=style, expand=False, title=title))
    log_messages.append(f"[{title}] {message}")


# Wake word configuration
wake_word = "nova"

# Initialize OpenAI client using environment configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required to run the assistant.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# System message for GPT models
sys_msg = (
    "You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context "
    "(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed "
    "text prompt that will be attached to their transcribed voice prompt. Generate the most useful and "
    "factual response possible, carefully considering all previous generated text in your response before "
    "adding new tokens to the response. Do not expect or request images, just use the context if added. "
    "Use all of the context of this conversation so your response is relevant to the conversation. Make "
    "your responses clear and concise, avoiding any verbosity."
)

convo: List[Dict[str, Any]] = [{"role": "system", "content": sys_msg}]

# Whisper configuration for on-device transcription
num_cores = max(os.cpu_count() or 2, 2)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=num_cores // 2, num_workers=num_cores // 2)

r = sr.Recognizer()


def save_log() -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for message in log_messages:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(f"Log saved to {filename}")


class EnhancedConversationContext:
    def __init__(self, max_turns: int = 5, similarity_threshold: float = 0.3) -> None:
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer()

    def add_exchange(self, user_input: str, assistant_response: str) -> None:
        if self.history:
            similarity = self.calculate_similarity(user_input)
            if similarity < self.similarity_threshold:
                self.clear()  # Clear context if topic changes significantly

        self.history.append({
            "user": user_input,
            "assistant": assistant_response,
        })
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self) -> str:
        if len(self.history) > 2:
            return self.summarize_context()
        return self.format_context()

    def format_context(self) -> str:
        context = ""
        for exchange in self.history:
            context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        return context.strip()

    def summarize_context(self) -> str:
        summary = "Previous conversation summary:\n"
        for exchange in self.history[:-1]:  # Summarize all but the last exchange
            summary += f"- User asked about {exchange['user'][:50]}...\n"
        summary += (
            "\nMost recent exchange:\n"
            f"User: {self.history[-1]['user']}\nAssistant: {self.history[-1]['assistant']}"
        )
        return summary

    def calculate_similarity(self, new_input: str) -> float:
        if not self.history:
            return 0.0
        previous_inputs = [exchange['user'] for exchange in self.history]
        previous_inputs.append(new_input)
        tfidf_matrix = self.vectorizer.fit_transform(previous_inputs)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        return float(np.mean(cosine_similarities))

    def clear(self) -> None:
        self.history = []

    def remember(self, information: str) -> None:
        self.history.append({"user": "Remember this", "assistant": information})

    def forget(self) -> str:
        self.clear()
        return "Previous context has been cleared."


# Initialize the enhanced conversation context
conversation_context = EnhancedConversationContext()


class ModelCatalog:
    """Maintain ordered preference lists for different OpenAI model capabilities."""

    def __init__(self, catalog: Dict[str, List[str]]) -> None:
        self._catalog = {capability: self._dedupe(models) for capability, models in catalog.items()}

    @staticmethod
    def _dedupe(models: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for model in models:
            if not model:
                continue
            if model not in seen:
                ordered.append(model)
                seen.add(model)
        return ordered

    def get(self, capability: str) -> List[str]:
        return self._catalog.get(capability, [])


class OpenAIModelManager:
    """Helper that automatically falls back between preferred models."""

    def __init__(self, client: OpenAI, catalog: ModelCatalog) -> None:
        self.client = client
        self.catalog = catalog

    def chat_completion(self, capability: str, **kwargs: Any):
        last_error: Optional[Exception] = None
        for model_name in self.catalog.get(capability):
            try:
                response = self.client.chat.completions.create(model=model_name, **kwargs)
                log(
                    f"Model '{model_name}' served the '{capability}' request.",
                    title="MODEL",
                    style="bold blue",
                )
                return response
            except Exception as exc:  # noqa: BLE001 - surface fallback errors
                last_error = exc
                log(
                    f"Model '{model_name}' unavailable ({exc}). Trying fallback...",
                    title="MODEL",
                    style="bold yellow",
                )
        raise RuntimeError(f"All models failed for capability '{capability}'.") from last_error


DEFAULT_MODEL_CATALOG = {
    "conversation": [
        os.getenv("OPENAI_PREFERRED_CHAT_MODEL", "gpt-5"),
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4o",
    ],
    "vision": [
        os.getenv("OPENAI_PREFERRED_VISION_MODEL", "gpt-5"),
        "gpt-5-mini",
        "gpt-4o",
    ],
    "structured": [
        os.getenv("OPENAI_PREFERRED_TOOL_MODEL", "gpt-5-mini"),
        "gpt-5-nano",
        "gpt-4o-mini",
        "gpt-4o",
    ],
}

model_manager = OpenAIModelManager(openai_client, ModelCatalog(DEFAULT_MODEL_CATALOG))


class ToolRegistry:
    """Registry for assistant tools exposed to OpenAI tool calling."""

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]],
        handler: Callable[..., str],
    ) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters if parameters is not None else {"type": "object", "properties": {}},
            "handler": handler,
        }

    def has_tools(self) -> bool:
        return bool(self._tools)

    def as_openai_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for tool in self._tools.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
            )
        return tools

    def execute(self, name: str, arguments_json: str) -> str:
        if name not in self._tools:
            log(f"Tool '{name}' is not registered.", title="TOOL", style="bold red")
            return f"Tool '{name}' is not available."

        handler = self._tools[name]["handler"]
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            if not isinstance(args, dict):
                args = {"value": args}
        except json.JSONDecodeError:
            args = {}

        log(f"Running tool '{name}' with args {args}.", title="TOOL", style="bold cyan")
        try:
            result = handler(**args)
        except Exception as exc:  # noqa: BLE001 - surface tool execution errors to the model
            log(f"Tool '{name}' failed: {exc}", title="TOOL", style="bold red")
            return f"Tool '{name}' execution failed: {exc}"

        if result is None:
            return ""
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)


tool_registry = ToolRegistry()


class ContextProvider:
    """Interface for external context providers (e.g., Model Context Protocol)."""

    def fetch_context(self, *, prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class ContextProviderRegistry:
    def __init__(self) -> None:
        self._providers: List[ContextProvider] = []

    def register(self, provider: ContextProvider) -> None:
        self._providers.append(provider)

    def gather(self, *, prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        contexts: List[str] = []
        for provider in self._providers:
            try:
                context = provider.fetch_context(prompt=prompt, conversation_history=conversation_history)
                if context:
                    contexts.append(context)
            except Exception as exc:  # noqa: BLE001 - providers should not crash the assistant
                log(f"Context provider error: {exc}", title="CONTEXT", style="bold yellow")
        return "\n\n".join(contexts)


class MCPContextProvider(ContextProvider):
    """Stub provider to integrate the Model Context Protocol (MCP)."""

    def __init__(self, endpoint: Optional[str] = None, default_namespace: Optional[str] = None) -> None:
        self.endpoint = endpoint or os.getenv("MCP_ENDPOINT")
        self.default_namespace = default_namespace or os.getenv("MCP_DEFAULT_NAMESPACE")
        self.context_file = os.getenv("MCP_CONTEXT_FILE")
        if self.context_file:
            log(
                f"MCP context file configured at {self.context_file}.",
                title="MCP",
                style="bold blue",
            )
        elif self.endpoint:
            log(
                "MCP endpoint configured. Implement custom client logic in MCPContextProvider.fetch_context.",
                title="MCP",
                style="bold blue",
            )

    def fetch_context(self, *, prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        if self.context_file:
            context_path = Path(self.context_file)
            if context_path.exists():
                try:
                    data = context_path.read_text(encoding="utf-8").strip()
                    if data:
                        return f"MCP context snippet:\n{data}"
                except Exception as exc:  # noqa: BLE001 - reading MCP file should not crash the assistant
                    log(f"Failed to read MCP context file: {exc}", title="MCP", style="bold yellow")
        # Placeholder for real MCP client integrations.
        return ""


context_provider_registry = ContextProviderRegistry()
context_provider_registry.register(MCPContextProvider())

ENABLE_TOOL_CALLING = os.getenv("ASSISTANT_DISABLE_TOOLS", "0").lower() not in {"1", "true", "yes"}
if not ENABLE_TOOL_CALLING:
    log("Tool calling disabled via ASSISTANT_DISABLE_TOOLS.", title="TOOLS", style="bold yellow")


def message_to_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if hasattr(message, "to_dict"):
        return message.to_dict()
    raise TypeError("Unsupported message type for serialization.")


def extract_message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    text_parts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            else:
                text_parts.append(str(part))
    return "".join(text_parts).strip()


def iter_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls = message.get("tool_calls") or []
    normalized: List[Dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            normalized.append(call)
        elif hasattr(call, "model_dump"):
            normalized.append(call.model_dump())
        elif hasattr(call, "to_dict"):
            normalized.append(call.to_dict())
    return normalized


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def take_screenshot() -> Optional[str]:
    log("Taking screenshot...", title="ACTION", style="bold blue")
    try:
        path = "screenshot.jpg"
        screenshot = ImageGrab.grab()
        rgb_screenshot = screenshot.convert("RGB")
        rgb_screenshot.save(path, quality=15)
        return path
    except Exception as exc:  # noqa: BLE001 - handle headless environments gracefully
        log(f"Error taking screenshot: {exc}", title="ERROR", style="bold red")
        return None


def web_cam_capture() -> Optional[str]:
    try:
        import pygame
        import pygame.camera

        pygame.camera.init()
        cameras = pygame.camera.list_cameras()

        if not cameras:
            log("Error: No cameras found", title="ERROR", style="bold red")
            return None

        cam = pygame.camera.Camera(cameras[0], (640, 480))
        cam.start()
        image = cam.get_image()
        cam.stop()
        pygame.camera.quit()

        path = "webcam.jpg"
        pygame.image.save(image, path)

        pil_string_image = pygame.image.tostring(image, "RGB", False)
        pil_image = Image.frombytes("RGB", (640, 480), pil_string_image)

        pil_image.save(path, "JPEG")

        log("Webcam image captured and saved.", title="ACTION", style="bold blue")
        return path
    except Exception as exc:  # noqa: BLE001 - webcam is optional
        log(f"Error capturing webcam image: {exc}", title="ERROR", style="bold red")
        return None


def get_clipboard_text() -> Optional[str]:
    log("Extracting clipboard text...", title="ACTION", style="bold blue")
    try:
        clipboard_content = pyperclip.paste()
    except pyperclip.PyperclipException as exc:
        log(f"Clipboard access failed: {exc}", title="ERROR", style="bold red")
        return None
    if isinstance(clipboard_content, str) and clipboard_content.strip():
        log("Clipboard text extracted.", title="ACTION", style="bold blue")
        return clipboard_content
    log("No clipboard text to copy", title="ERROR", style="bold red")
    return None


def vision_prompt(prompt: str, photo_path: str) -> str:
    log("Generating vision prompt...", title="ACTION", style="bold blue")
    try:
        encoded_image = encode_image(photo_path)
    except FileNotFoundError:
        log("Image file not found for vision prompt.", title="ERROR", style="bold red")
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image in the context of the following prompt: "
                        f"{prompt}. Provide a detailed description focusing on elements relevant to the prompt."
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encoded_image}", "detail": "high"}},
            ],
        }
    ]
    try:
        response = model_manager.chat_completion(
            "vision",
            messages=messages,
            max_tokens=300,
        )
    except Exception as exc:  # noqa: BLE001 - surface the failure but keep running
        log(f"Vision model failed: {exc}", title="ERROR", style="bold red")
        return ""
    message_dict = message_to_dict(response.choices[0].message)
    description = extract_message_text(message_dict)
    log("Vision prompt generated.", title="ACTION", style="bold blue")
    return description


def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as exc:  # noqa: BLE001 - internet search is optional
        log(f"Error in DuckDuckGo search: {exc}", title="ERROR", style="bold red")
        return []


def process_search_results(results: List[Dict[str, Any]]) -> str:
    processed = "Search results:\n\n"
    for i, result in enumerate(results, 1):
        processed += f"{i}. {result['title']}\n   {result['body']}\n   URL: {result['href']}\n\n"
    return processed.strip()


def capture_screenshot_context_tool(user_prompt: str = "") -> str:
    photo_path = take_screenshot()
    if not photo_path:
        return "Screenshot capture failed."
    description = vision_prompt(user_prompt or "Describe this screenshot for the assistant.", photo_path)
    if not description:
        description = "No description available."
    return f"Screenshot stored at {photo_path}. Description: {description}"


def capture_webcam_context_tool(user_prompt: str = "") -> str:
    photo_path = web_cam_capture()
    if not photo_path:
        return "Webcam capture failed."
    description = vision_prompt(user_prompt or "Describe this webcam image for the assistant.", photo_path)
    if not description:
        description = "No description available."
    return f"Webcam image stored at {photo_path}. Description: {description}"


def extract_clipboard_text_tool() -> str:
    text = get_clipboard_text()
    if text:
        return text
    return "Clipboard is empty or unavailable."


def duckduckgo_search_tool(query: str, max_results: int = 5) -> str:
    results = duckduckgo_search(query, max_results=max_results)
    if not results:
        return "No DuckDuckGo search results found."
    return process_search_results(results)


def register_builtin_tools() -> None:
    tool_registry.register(
        name="extract_clipboard_text",
        description="Extract the latest textual content from the user's clipboard.",
        parameters={"type": "object", "properties": {}},
        handler=lambda: extract_clipboard_text_tool(),
    )
    tool_registry.register(
        name="capture_screenshot_context",
        description=(
            "Capture a screenshot on the user's machine (macOS supported) and describe it for additional conversation context."
        ),
        parameters={
            "type": "object",
            "properties": {
                "user_prompt": {
                    "type": "string",
                    "description": "The user's current request to guide the screenshot analysis.",
                }
            },
        },
        handler=lambda user_prompt="": capture_screenshot_context_tool(user_prompt=user_prompt),
    )
    tool_registry.register(
        name="capture_webcam_context",
        description="Capture a webcam photo and describe it for additional conversation context.",
        parameters={
            "type": "object",
            "properties": {
                "user_prompt": {
                    "type": "string",
                    "description": "The user's current request to guide the webcam analysis.",
                }
            },
        },
        handler=lambda user_prompt="": capture_webcam_context_tool(user_prompt=user_prompt),
    )
    tool_registry.register(
        name="duckduckgo_search",
        description="Perform a DuckDuckGo search and return the most relevant results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to run on DuckDuckGo.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
        handler=lambda query, max_results=5: duckduckgo_search_tool(query=query, max_results=max_results),
    )


register_builtin_tools()


def complete_chat_with_tools(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    while True:
        params: Dict[str, Any] = {"messages": messages}
        tools_enabled = ENABLE_TOOL_CALLING and tool_registry.has_tools()
        if tools_enabled:
            params["tools"] = tool_registry.as_openai_tools()
            params["tool_choice"] = "auto"

        response = model_manager.chat_completion("conversation", **params)
        message_dict = message_to_dict(response.choices[0].message)

        tool_calls = iter_tool_calls(message_dict) if tools_enabled else []
        if tool_calls:
            messages.append(message_dict)
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments", "{}")
                result = tool_registry.execute(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": result,
                    }
                )
            continue

        messages.append(message_dict)
        return message_dict


def llm_prompt(prompt: str, img_context: Optional[str]) -> str:
    base_prompt = prompt
    context = conversation_context.get_context()
    provider_context = context_provider_registry.gather(
        prompt=prompt, conversation_history=conversation_context.history
    )

    if context:
        prompt = f"Previous conversation:\n{context}\n\nCurrent user prompt: {prompt}"
    if provider_context:
        prompt = f"{prompt}\n\nAdditional context from providers:\n{provider_context}"
    if img_context:
        prompt = f"{prompt}\n\nIMAGE CONTEXT: {img_context}"

    convo.append({"role": "user", "content": prompt})
    response_message = complete_chat_with_tools(convo)
    response_text = extract_message_text(response_message)

    conversation_context.add_exchange(base_prompt, response_text)
    return response_text


def speak(text: str) -> None:
    player = pyaudio.PyAudio()
    player_stream = player.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_started = False
    with openai_client.audio.speech.with_streaming_response.create(
        model="tts-1", voice="nova", response_format="pcm", speed="1.75", input=text
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_started:
                player_stream.write(chunk)
            else:
                if chunk and max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_started = True


def wav_to_text(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(audio_path)
    text = "".join(segment.text for segment in segments)
    return text


def extract_prompt(transcribed_text: str, wake_word_value: str) -> Optional[str]:
    pattern = rf"\b{re.escape(wake_word_value)}[\s,.?!]*(.*)"
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    if match:
        prompt = match.group(1).strip()
        return prompt
    return None


def callback(recognizer, audio) -> None:  # noqa: ANN001 - callback signature defined by SpeechRecognition
    prompt_audio_path = "prompt.wav"
    with open(prompt_audio_path, "wb") as f:
        f.write(audio.get_wav_data())
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)
    if clean_prompt:
        log(f"USER: {clean_prompt}", title="USER INPUT", style="bold green")

        if clean_prompt.lower().startswith("remember "):
            conversation_context.remember(clean_prompt[9:])
            response = "I've remembered that information."
        elif clean_prompt.lower() == "forget context":
            response = conversation_context.forget()
        elif clean_prompt.lower().startswith("search "):
            search_query = clean_prompt[7:]
            search_results = duckduckgo_search(search_query)
            processed_results = process_search_results(search_results)
            response = llm_prompt(
                prompt=(
                    "Based on the following search results, answer the query: "
                    f"{search_query}\n\n{processed_results}"
                ),
                img_context=None,
            )
        else:
            response = llm_prompt(prompt=clean_prompt, img_context=None)

        log(f"ASSISTANT: {response}", title="ASSISTANT RESPONSE", style="bold magenta")
        speak(response)


def start_listening() -> None:
    log("Adjusting for ambient noise...", title="ACTION", style="bold blue")
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=2)
        console.print(Panel("Say 'nova' followed with your prompt.", border_style="bold magenta", title="INSTRUCTIONS"))
    stop_listening = r.listen_in_background(sr.Microphone(), callback)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)
        log("Listening stopped.", title="ACTION", style="bold blue")
        save_log()


if __name__ == "__main__":
    start_listening()
