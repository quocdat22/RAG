"""
Chat History Management for persistent conversation storage.

This module provides functionality to save, load, and manage
conversation histories as JSON files.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logging import LoggerMixin
from src.core.memory import ConversationMemory, ConversationTurn


@dataclass
class ConversationMetadata:
    """Metadata for a saved conversation."""
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SavedConversation:
    """A saved conversation with metadata and history."""
    metadata: ConversationMetadata
    history: list[dict[str, Any]]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "history": self.history,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SavedConversation":
        """Create from dictionary."""
        return cls(
            metadata=ConversationMetadata.from_dict(data["metadata"]),
            history=data["history"],
        )


class ChatHistoryManager(LoggerMixin):
    """
    Manages persistent chat history storage.
    
    Features:
    - Save/load conversations as JSON files
    - List all saved conversations
    - Delete conversations
    - Export conversations to markdown
    """
    
    def __init__(self, storage_dir: Path, max_conversations: int = 100):
        """
        Initialize chat history manager.
        
        Args:
            storage_dir: Directory to store conversation files
            max_conversations: Maximum number of conversations to retain
        """
        self.storage_dir = Path(storage_dir)
        self.max_conversations = max_conversations
        
        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            f"ChatHistoryManager initialized: storage_dir={storage_dir}, "
            f"max_conversations={max_conversations}"
        )
    
    def _get_conversation_path(self, conversation_id: str) -> Path:
        """Get path to conversation file."""
        return self.storage_dir / f"{conversation_id}.json"
    
    def _generate_title(self, first_query: str, max_length: int = 50) -> str:
        """Generate conversation title from first query."""
        if len(first_query) <= max_length:
            return first_query
        return first_query[:max_length - 3] + "..."
    
    def save_conversation(
        self,
        memory: ConversationMemory,
        conversation_id: str | None = None,
        title: str | None = None,
    ) -> str:
        """
        Save a conversation from memory.
        
        Args:
            memory: ConversationMemory to save
            conversation_id: Optional conversation ID (uses memory.session_id if None)
            title: Optional custom title
            
        Returns:
            Conversation ID
        """
        history = memory.get_history()
        if not history:
            self.logger.warning("Cannot save empty conversation")
            return ""
        
        conversation_id = conversation_id or memory.session_id
        
        # Generate title if not provided
        if title is None:
            first_query = history[0]["query"]
            title = self._generate_title(first_query)
        
        # Check if conversation exists
        conversation_path = self._get_conversation_path(conversation_id)
        if conversation_path.exists():
            # Load existing to preserve created_at
            existing = self.load_conversation(conversation_id)
            created_at = existing.metadata.created_at if existing else datetime.now().isoformat()
        else:
            created_at = datetime.now().isoformat()
        
        # Create conversation data
        metadata = ConversationMetadata(
            conversation_id=conversation_id,
            title=title,
            created_at=created_at,
            updated_at=datetime.now().isoformat(),
            turn_count=len(history),
        )
        
        conversation = SavedConversation(
            metadata=metadata,
            history=history,
        )
        
        # Save to file
        try:
            with open(conversation_path, "w", encoding="utf-8") as f:
                json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved conversation {conversation_id} with {len(history)} turns")
            
            # Clean up old conversations if needed
            self._cleanup_old_conversations()
            
            return conversation_id
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation {conversation_id}: {e}")
            raise
    
    def load_conversation(self, conversation_id: str) -> SavedConversation | None:
        """
        Load a saved conversation.
        
        Args:
            conversation_id: ID of conversation to load
            
        Returns:
            SavedConversation or None if not found
        """
        conversation_path = self._get_conversation_path(conversation_id)
        
        if not conversation_path.exists():
            self.logger.warning(f"Conversation {conversation_id} not found")
            return None
        
        try:
            with open(conversation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            conversation = SavedConversation.from_dict(data)
            self.logger.info(f"Loaded conversation {conversation_id}")
            return conversation
            
        except Exception as e:
            self.logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return None
    
    def restore_to_memory(
        self,
        conversation_id: str,
        max_turns: int = 10,
        max_tokens: int = 4000,
    ) -> ConversationMemory | None:
        """
        Load conversation and restore to ConversationMemory.
        
        Args:
            conversation_id: ID of conversation to restore
            max_turns: Max turns for memory
            max_tokens: Max tokens for memory
            
        Returns:
            ConversationMemory with restored history, or None if not found
        """
        conversation = self.load_conversation(conversation_id)
        if not conversation:
            return None
        
        # Create new memory
        memory = ConversationMemory(
            max_turns=max_turns,
            max_tokens=max_tokens,
            session_id=conversation_id,
        )
        
        # Restore turns
        for turn_data in conversation.history:
            # Reconstruct turn (skip timestamp to use current time)
            memory.add_turn(
                query=turn_data["query"],
                answer=turn_data["answer"],
                sources=turn_data.get("sources", []),
                metadata=turn_data.get("metadata", {}),
            )
        
        self.logger.info(
            f"Restored conversation {conversation_id} to memory with "
            f"{len(conversation.history)} turns"
        )
        return memory
    
    def list_conversations(self, limit: int | None = None) -> list[ConversationMetadata]:
        """
        List all saved conversations.
        
        Args:
            limit: Optional limit on number of conversations to return
            
        Returns:
            List of conversation metadata, sorted by updated_at (newest first)
        """
        conversations = []
        
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                metadata = ConversationMetadata.from_dict(data["metadata"])
                conversations.append(metadata)
                
            except Exception as e:
                self.logger.warning(f"Failed to load metadata from {path}: {e}")
                continue
        
        # Sort by updated_at (newest first)
        conversations.sort(key=lambda x: x.updated_at, reverse=True)
        
        if limit:
            conversations = conversations[:limit]
        
        self.logger.debug(f"Listed {len(conversations)} conversations")
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a saved conversation.
        
        Args:
            conversation_id: ID of conversation to delete
            
        Returns:
            True if deleted, False if not found
        """
        conversation_path = self._get_conversation_path(conversation_id)
        
        if not conversation_path.exists():
            self.logger.warning(f"Conversation {conversation_id} not found for deletion")
            return False
        
        try:
            conversation_path.unlink()
            self.logger.info(f"Deleted conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            return False
    
    def rename_conversation(self, conversation_id: str, new_title: str) -> bool:
        """
        Rename a saved conversation.
        
        Args:
            conversation_id: ID of conversation to rename
            new_title: New title for the conversation
            
        Returns:
            True if renamed, False if not found
        """
        conversation = self.load_conversation(conversation_id)
        if not conversation:
            return False
        
        # Update title and save
        conversation.metadata.title = new_title
        conversation.metadata.updated_at = datetime.now().isoformat()
        
        try:
            conversation_path = self._get_conversation_path(conversation_id)
            with open(conversation_path, "w", encoding="utf-8") as f:
                json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Renamed conversation {conversation_id} to '{new_title}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rename conversation {conversation_id}: {e}")
            return False
    
    def export_to_markdown(self, conversation_id: str) -> str | None:
        """
        Export conversation to markdown format.
        
        Args:
            conversation_id: ID of conversation to export
            
        Returns:
            Markdown string or None if not found
        """
        conversation = self.load_conversation(conversation_id)
        if not conversation:
            return None
        
        lines = [
            f"# {conversation.metadata.title}",
            "",
            f"**Created:** {conversation.metadata.created_at}",
            f"**Updated:** {conversation.metadata.updated_at}",
            f"**Turns:** {conversation.metadata.turn_count}",
            "",
            "---",
            "",
        ]
        
        for i, turn in enumerate(conversation.history, 1):
            lines.extend([
                f"## Turn {i}",
                "",
                f"**User:** {turn['query']}",
                "",
                f"**Assistant:** {turn['answer']}",
                "",
            ])
            
            if turn.get("sources"):
                lines.append("**Sources:**")
                for source in turn["sources"]:
                    lines.append(f"- {source}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _cleanup_old_conversations(self) -> None:
        """Remove oldest conversations if limit exceeded."""
        conversations = self.list_conversations()
        
        if len(conversations) <= self.max_conversations:
            return
        
        # Delete oldest conversations
        to_delete = conversations[self.max_conversations:]
        for metadata in to_delete:
            self.delete_conversation(metadata.conversation_id)
        
        self.logger.info(
            f"Cleaned up {len(to_delete)} old conversations "
            f"(limit: {self.max_conversations})"
        )


__all__ = [
    "ChatHistoryManager",
    "ConversationMetadata",
    "SavedConversation",
]
