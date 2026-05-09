"""Core orchestration logic for the voice assistant."""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional

import speech_recognition as sr
from rich.panel import Panel

from .config import (
    WAKE_WORD,
    SYSTEM_MESSAGE,
    ENABLE_TOOL_CALLING,
    SIMPLE_TOOLS,
    TTS_PROVIDER,
)
from .context import EnhancedConversationContext, ContextProviderRegistry, MCPContextProvider
from .providers.llm import get_llm_provider, LLMProvider
from .providers.tts import get_tts_provider, TTSProvider
from .speech import wav_to_text, extract_prompt
from .tools import (
    ToolLoop,
    ToolRegistry,
    capture_screenshot_context_tool,
    capture_webcam_context_tool,
    duckduckgo_search,
    duckduckgo_search_tool,
    extract_clipboard_text_tool,
    process_search_results,
)
from .tools.vision_tools import set_llm_provider
from .utils import console, log, save_log


class VoiceAssistant:
    """Main voice assistant orchestrator."""

    def __init__(self) -> None:
        self.llm_provider: LLMProvider = get_llm_provider()
        self.tts_provider: TTSProvider = get_tts_provider()
        set_llm_provider(self.llm_provider)

        self.conversation_context = EnhancedConversationContext()
        self.context_provider_registry = ContextProviderRegistry()
        self.context_provider_registry.register(MCPContextProvider())

        self.tool_registry = ToolRegistry()
        self._register_builtin_tools()

        self.tool_loop = ToolLoop(
            llm_provider=self.llm_provider,
            tool_registry=self.tool_registry,
            tools_enabled=ENABLE_TOOL_CALLING,
        )

        self.convo: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_MESSAGE}]
        self.recognizer = sr.Recognizer()

        if not ENABLE_TOOL_CALLING:
            log("Tool calling disabled via ASSISTANT_DISABLE_TOOLS.", title="TOOLS", style="bold yellow")
        if SIMPLE_TOOLS:
            log("Simple tools mode enabled. Vision tools disabled.", title="TOOLS", style="bold blue")

    def _register_builtin_tools(self) -> None:
        """Register built-in tools with the registry."""
        self.tool_registry.register(
            name="extract_clipboard_text",
            description="Extract the latest textual content from the user's clipboard.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: extract_clipboard_text_tool(),
        )

        self.tool_registry.register(
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

        if SIMPLE_TOOLS:
            return

        self.tool_registry.register(
            name="capture_screenshot_context",
            description=(
                "Capture a screenshot on the user's machine (macOS supported) and "
                "describe it for additional conversation context."
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
        self.tool_registry.register(
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

    def llm_prompt(self, prompt: str, img_context: Optional[str] = None) -> str:
        """Run the user's prompt through the LLM and stream the response to TTS."""
        base_prompt = prompt
        context = self.conversation_context.get_context()
        provider_context = self.context_provider_registry.gather(
            prompt=prompt, conversation_history=self.conversation_context.history
        )

        if context:
            prompt = f"Previous conversation:\n{context}\n\nCurrent user prompt: {prompt}"
        if provider_context:
            prompt = f"{prompt}\n\nAdditional context from providers:\n{provider_context}"
        if img_context:
            prompt = f"{prompt}\n\nIMAGE CONTEXT: {img_context}"

        self.convo.append({"role": "user", "content": prompt})

        full_response_text = ""

        def speakable_chunks():
            nonlocal full_response_text
            for chunk in self.tool_loop.stream(self.convo):
                full_response_text += chunk
                yield chunk

        self.tts_provider.stream_speak(speakable_chunks())

        self.conversation_context.add_exchange(base_prompt, full_response_text)
        return full_response_text

    def speak(self, text: str) -> None:
        """Speak text using the configured TTS provider with fallback to OpenAI."""
        if self.tts_provider.speak(text):
            return

        if TTS_PROVIDER != "openai":
            log("Falling back to OpenAI TTS.", title="TTS", style="bold yellow")
            from .providers.tts.openai_tts import OpenAITTSProvider
            try:
                fallback = OpenAITTSProvider()
                if fallback.speak(text):
                    return
            except Exception:  # noqa: BLE001 - fallback is best-effort
                pass

        log("Unable to synthesise speech for the assistant response.", title="TTS", style="bold red")

    def callback(self, recognizer: sr.Recognizer, audio: sr.AudioData) -> None:
        """Audio callback for background listening."""
        wav_data = io.BytesIO(audio.get_wav_data())
        prompt_text = wav_to_text(wav_data)
        log(f"Heard: {prompt_text!r}", title="DEBUG", style="dim")
        clean_prompt = extract_prompt(prompt_text, WAKE_WORD)

        if not clean_prompt:
            return

        log(f"USER: {clean_prompt}", title="USER INPUT", style="bold green")
        response = self._handle_command(clean_prompt)
        log(f"ASSISTANT: {response}", title="ASSISTANT RESPONSE", style="bold magenta")

    def _handle_command(self, clean_prompt: str) -> str:
        """Route a recognised prompt to memory commands, search, or the LLM."""
        lowered = clean_prompt.lower()
        if lowered.startswith("remember "):
            self.conversation_context.remember(clean_prompt[9:])
            return "I've remembered that information."
        if lowered == "forget context":
            return self.conversation_context.forget()
        if lowered.startswith("search "):
            search_query = clean_prompt[7:]
            search_results = duckduckgo_search(search_query)
            processed_results = process_search_results(search_results)
            return self.llm_prompt(
                prompt=(
                    "Based on the following search results, answer the query: "
                    f"{search_query}\n\n{processed_results}"
                ),
                img_context=None,
            )
        return self.llm_prompt(prompt=clean_prompt, img_context=None)

    def start_listening(self) -> None:
        """Start the background listening loop."""
        log("Adjusting for ambient noise...", title="ACTION", style="bold blue")
        self.recognizer.pause_threshold = 1.5
        self.recognizer.phrase_threshold = 0.3
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            console.print(
                Panel(
                    f"Say '{WAKE_WORD}' followed with your prompt.",
                    border_style="bold magenta",
                    title="INSTRUCTIONS",
                )
            )

        stop_listening = self.recognizer.listen_in_background(sr.Microphone(), self.callback)
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            stop_listening(wait_for_stop=False)
            log("Listening stopped.", title="ACTION", style="bold blue")
            save_log()


def main() -> None:
    """Main entry point for the voice assistant."""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="faster_whisper")

    assistant = VoiceAssistant()
    assistant.start_listening()


if __name__ == "__main__":
    main()
