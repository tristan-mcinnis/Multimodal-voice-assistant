"""Web search tool implementations."""

from __future__ import annotations

from typing import Any, Dict, List

from duckduckgo_search import DDGS

from ..utils import log


def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Perform a DuckDuckGo search.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        List of search result dictionaries
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as exc:  # noqa: BLE001 - internet search is optional
        log(f"Error in DuckDuckGo search: {exc}", title="ERROR", style="bold red")
        return []


def process_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results for the assistant.

    Args:
        results: List of search result dictionaries

    Returns:
        Formatted string of search results
    """
    processed = "Search results:\n\n"
    for i, result in enumerate(results, 1):
        processed += f"{i}. {result['title']}\n   {result['body']}\n   URL: {result['href']}\n\n"
    return processed.strip()


def duckduckgo_search_tool(query: str, max_results: int = 5) -> str:
    """Tool handler for DuckDuckGo search.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Formatted search results or error message
    """
    results = duckduckgo_search(query, max_results=max_results)
    if not results:
        return "No DuckDuckGo search results found."
    return process_search_results(results)
