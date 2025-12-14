"""Conversation context management with TF-IDF similarity-based topic detection."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EnhancedConversationContext:
    """Manages conversation history with automatic topic change detection."""

    def __init__(self, max_turns: int = 5, similarity_threshold: float = 0.3) -> None:
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer()

    def add_exchange(self, user_input: str, assistant_response: str) -> None:
        """Add a conversation exchange, clearing context if topic changes significantly."""
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
        """Get formatted context, summarizing if conversation is long."""
        if len(self.history) > 2:
            return self.summarize_context()
        return self.format_context()

    def format_context(self) -> str:
        """Format the full conversation context."""
        context = ""
        for exchange in self.history:
            context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        return context.strip()

    def summarize_context(self) -> str:
        """Create a summarized version of the conversation context."""
        summary = "Previous conversation summary:\n"
        for exchange in self.history[:-1]:  # Summarize all but the last exchange
            summary += f"- User asked about {exchange['user'][:50]}...\n"
        summary += (
            "\nMost recent exchange:\n"
            f"User: {self.history[-1]['user']}\nAssistant: {self.history[-1]['assistant']}"
        )
        return summary

    def calculate_similarity(self, new_input: str) -> float:
        """Calculate similarity between new input and conversation history."""
        if not self.history:
            return 0.0
        previous_inputs = [exchange['user'] for exchange in self.history]
        previous_inputs.append(new_input)
        tfidf_matrix = self.vectorizer.fit_transform(previous_inputs)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        return float(np.mean(cosine_similarities))

    def clear(self) -> None:
        """Clear all conversation history."""
        self.history = []

    def remember(self, information: str) -> None:
        """Store information in the conversation context."""
        self.history.append({"user": "Remember this", "assistant": information})

    def forget(self) -> str:
        """Clear context and return confirmation message."""
        self.clear()
        return "Previous context has been cleared."
