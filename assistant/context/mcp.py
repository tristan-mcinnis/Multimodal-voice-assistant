"""Model Context Protocol (MCP) integration for external context providers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..config import MCP_CONTEXT_FILE, MCP_ENDPOINT, MCP_DEFAULT_NAMESPACE
from ..utils import log


class ContextProvider:
    """Interface for external context providers."""

    def fetch_context(self, *, prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        """Fetch context relevant to the current prompt and conversation."""
        raise NotImplementedError


class ContextProviderRegistry:
    """Registry for managing multiple context providers."""

    def __init__(self) -> None:
        self._providers: List[ContextProvider] = []

    def register(self, provider: ContextProvider) -> None:
        """Register a new context provider."""
        self._providers.append(provider)

    def gather(self, *, prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        """Gather context from all registered providers."""
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

    def __init__(
        self,
        endpoint: Optional[str] = None,
        default_namespace: Optional[str] = None,
        context_file: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint or MCP_ENDPOINT
        self.default_namespace = default_namespace or MCP_DEFAULT_NAMESPACE
        self.context_file = context_file or MCP_CONTEXT_FILE

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
        """Fetch context from file or MCP endpoint."""
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
