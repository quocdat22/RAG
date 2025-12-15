"""
Conversation Memory for multi-turn chat.

This module manages conversation history to enable
context-aware responses in multi-turn conversations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.core.logging import LoggerMixin
from src.core.utils import count_tokens


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    query: str
    answer: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
            "sources": self.sources,
            "metadata": self.metadata,
        }


class ConversationMemory(LoggerMixin):
    """
    Manages conversation history for multi-turn chat.
    
    Features:
    - Sliding window to limit context size
    - Token counting to stay within limits
    - Formatted context generation for LLM
    """
    
    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 4000,
        session_id: str | None = None,
    ):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of turns to keep
            max_tokens: Maximum tokens for context
            session_id: Optional session identifier
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.session_id = session_id or self._generate_session_id()
        
        self._history: list[ConversationTurn] = []
        
        self.logger.info(
            f"ConversationMemory initialized: session={self.session_id}, "
            f"max_turns={max_turns}, max_tokens={max_tokens}"
        )
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def add_turn(
        self,
        query: str,
        answer: str,
        sources: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a conversation turn.
        
        Args:
            query: User query
            answer: System answer
            sources: Source documents used
            metadata: Additional metadata
        """
        turn = ConversationTurn(
            query=query,
            answer=answer,
            sources=sources or [],
            metadata=metadata or {},
        )
        
        self._history.append(turn)
        
        # Trim to max turns
        if len(self._history) > self.max_turns:
            removed = len(self._history) - self.max_turns
            self._history = self._history[-self.max_turns:]
            self.logger.debug(f"Trimmed {removed} old turns from memory")
        
        self.logger.debug(f"Added turn to memory (total: {len(self._history)})")
    
    def get_context(self, max_tokens: int | None = None) -> str:
        """
        Get formatted conversation context for LLM.
        
        Args:
            max_tokens: Override max tokens limit
            
        Returns:
            Formatted conversation history
        """
        if not self._history:
            return ""
        
        max_tokens = max_tokens or self.max_tokens
        
        # Build context from recent history
        context_parts = []
        total_tokens = 0
        
        # Process from most recent backward
        for turn in reversed(self._history):
            turn_text = f"User: {turn.query}\nAssistant: {turn.answer}"
            turn_tokens = count_tokens(turn_text)
            
            if total_tokens + turn_tokens > max_tokens:
                break
            
            context_parts.insert(0, turn_text)
            total_tokens += turn_tokens
        
        if not context_parts:
            return ""
        
        context = "Previous conversation:\n" + "\n\n".join(context_parts)
        
        self.logger.debug(
            f"Generated context: {len(context_parts)} turns, {total_tokens} tokens"
        )
        
        return context
    
    def get_last_query(self) -> str | None:
        """Get the last user query."""
        if self._history:
            return self._history[-1].query
        return None
    
    def get_last_answer(self) -> str | None:
        """Get the last system answer."""
        if self._history:
            return self._history[-1].answer
        return None
    
    def get_history(self) -> list[dict[str, Any]]:
        """Get full conversation history as list of dicts."""
        return [turn.to_dict() for turn in self._history]
    
    def get_turn_count(self) -> int:
        """Get number of turns in memory."""
        return len(self._history)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self._history = []
        self.logger.info(f"Cleared memory for session {self.session_id}")
    
    def summarize(self) -> str:
        """
        Get a brief summary of the conversation.
        
        Returns:
            Summary string
        """
        if not self._history:
            return "No conversation history."
        
        first_query = self._history[0].query
        last_query = self._history[-1].query
        
        return (
            f"Conversation with {len(self._history)} turns. "
            f"Started with: '{first_query[:50]}...' "
            f"Last query: '{last_query[:50]}...'"
        )


# Session storage for multiple conversations
_sessions: dict[str, ConversationMemory] = {}


def get_or_create_memory(
    session_id: str,
    max_turns: int = 10,
    max_tokens: int = 4000,
) -> ConversationMemory:
    """
    Get or create a conversation memory for a session.
    
    Args:
        session_id: Session identifier
        max_turns: Maximum turns to keep
        max_tokens: Maximum context tokens
        
    Returns:
        ConversationMemory instance
    """
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(
            max_turns=max_turns,
            max_tokens=max_tokens,
            session_id=session_id,
        )
    return _sessions[session_id]


def clear_session(session_id: str) -> bool:
    """
    Clear and remove a session's memory.
    
    Args:
        session_id: Session identifier
        
    Returns:
        True if session existed and was cleared
    """
    if session_id in _sessions:
        _sessions[session_id].clear()
        del _sessions[session_id]
        return True
    return False


__all__ = [
    "ConversationMemory",
    "ConversationTurn",
    "get_or_create_memory",
    "clear_session",
]
