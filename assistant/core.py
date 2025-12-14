"""Core orchestration logic for the voice assistant."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import speech_recognition as sr

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
    ToolRegistry,
    capture_screenshot_context_tool,
    capture_webcam_context_tool,
    extract_clipboard_text_tool,
    duckduckgo_search_tool,
    duckduckgo_search,
    process_search_results,
)
from .tools.vision_tools import set_llm_provider
from .utils import log, save_log, console, message_to_dict, extract_message_text, iter_tool_calls

from rich.panel import Panel


class VoiceAssistant:
    """Main voice assistant orchestrator."""

    def __init__(self) -> None:
        # Initialize providers
        self.llm_provider: LLMProvider = get_llm_provider()
        self.tts_provider: TTSProvider = get_tts_provider()

        # Set up vision tools with LLM provider
        set_llm_provider(self.llm_provider)

        # Initialize context
        self.conversation_context = EnhancedConversationContext()
        self.context_provider_registry = ContextProviderRegistry()
        self.context_provider_registry.register(MCPContextProvider())

        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        self._register_builtin_tools()

        # Conversation history
        self.convo: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_MESSAGE}]

        # Speech recognition
        self.recognizer = sr.Recognizer()

        # Log configuration
        if not ENABLE_TOOL_CALLING:
            log("Tool calling disabled via ASSISTANT_DISABLE_TOOLS.", title="TOOLS", style="bold yellow")
        if SIMPLE_TOOLS:
            log("Simple tools mode enabled. Vision tools disabled.", title="TOOLS", style="bold blue")

    def _register_builtin_tools(self) -> None:
        """Register built-in tools with the registry."""
        # Always register clipboard tool - works locally
        self.tool_registry.register(
            name="extract_clipboard_text",
            description="Extract the latest textual content from the user's clipboard.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: extract_clipboard_text_tool(),
        )

        # Always register web search - works locally
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

        # Vision-heavy tools - only register if not in simple tools mode
        if not SIMPLE_TOOLS:
            self.tool_registry.register(
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

    def complete_chat_with_tools(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute chat completion with tool calling loop.

        Args:
            messages: The conversation messages

        Returns:
            The final assistant message
        """
        while True:
            params: Dict[str, Any] = {"messages": messages}
            tools_enabled = ENABLE_TOOL_CALLING and self.tool_registry.has_tools()
            if tools_enabled:
                params["tools"] = self.tool_registry.as_openai_tools()
                params["tool_choice"] = "auto"

            response = self.llm_provider.chat_completion("conversation", **params)
            message_dict = message_to_dict(response.choices[0].message)

            tool_calls = iter_tool_calls(message_dict) if tools_enabled else []
            if tool_calls:
                messages.append(message_dict)
                for call in tool_calls:
                    function = call.get("function", {})
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                    result = self.tool_registry.execute(name, arguments)
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

    def llm_prompt(self, prompt: str, img_context: Optional[str] = None) -> str:
        """Process a prompt through the LLM.

        Args:
            prompt: The user's prompt
            img_context: Optional image context

        Returns:
            The assistant's response
        """
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
        response_message = self.complete_chat_with_tools(self.convo)
        response_text = extract_message_text(response_message)

        self.conversation_context.add_exchange(base_prompt, response_text)
        return response_text

    def speak(self, text: str) -> None:
        """Speak text using the configured TTS provider with fallback.

        Args:
            text: The text to speak
        """
        if self.tts_provider.speak(text):
            return

        # Fallback to OpenAI TTS if primary provider fails
        if TTS_PROVIDER != "openai":
            log("Falling back to OpenAI TTS.", title="TTS", style="bold yellow")
            from .providers.tts.openai_tts import OpenAITTSProvider
            try:
                fallback = OpenAITTSProvider()
                if fallback.speak(text):
                    return
            except Exception:
                pass

        log("Unable to synthesise speech for the assistant response.", title="TTS", style="bold red")

    def callback(self, recognizer: sr.Recognizer, audio: sr.AudioData) -> None:
        """Audio callback for background listening.

        Args:
            recognizer: The speech recognizer
            audio: The audio data
        """
        prompt_audio_path = "prompt.wav"
        with open(prompt_audio_path, "wb") as f:
            f.write(audio.get_wav_data())
        prompt_text = wav_to_text(prompt_audio_path)
        clean_prompt = extract_prompt(prompt_text, WAKE_WORD)

        if clean_prompt:
            log(f"USER: {clean_prompt}", title="USER INPUT", style="bold green")

            if clean_prompt.lower().startswith("remember "):
                self.conversation_context.remember(clean_prompt[9:])
                response = "I've remembered that information."
            elif clean_prompt.lower() == "forget context":
                response = self.conversation_context.forget()
            elif clean_prompt.lower().startswith("search "):
                search_query = clean_prompt[7:]
                search_results = duckduckgo_search(search_query)
                processed_results = process_search_results(search_results)
                response = self.llm_prompt(
                    prompt=(
                        "Based on the following search results, answer the query: "
                        f"{search_query}\n\n{processed_results}"
                    ),
                    img_context=None,
                )
            else:
                response = self.llm_prompt(prompt=clean_prompt, img_context=None)

            log(f"ASSISTANT: {response}", title="ASSISTANT RESPONSE", style="bold magenta")
            self.speak(response)

    def start_listening(self) -> None:
        """Start the background listening loop."""
        log("Adjusting for ambient noise...", title="ACTION", style="bold blue")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            console.print(Panel(f"Say '{WAKE_WORD}' followed with your prompt.", border_style="bold magenta", title="INSTRUCTIONS"))

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
