"""
Unit tests for ChatHistoryManager.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from src.core.memory import ConversationMemory
from src.core.chat_history import ChatHistoryManager, ConversationMetadata, SavedConversation


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "conversations"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def manager(temp_storage):
    """Create ChatHistoryManager instance."""
    return ChatHistoryManager(storage_dir=temp_storage, max_conversations=5)


@pytest.fixture
def sample_memory():
    """Create sample conversation memory."""
    memory = ConversationMemory(max_turns=10, max_tokens=4000)
    memory.add_turn(
        query="What is machine learning?",
        answer="Machine learning is a subset of AI...",
        sources=["paper1.pdf", "paper2.pdf"],
    )
    memory.add_turn(
        query="What are neural networks?",
        answer="Neural networks are computing systems...",
        sources=["paper3.pdf"],
    )
    return memory


def test_save_conversation(manager, sample_memory):
    """Test saving a conversation."""
    conv_id = manager.save_conversation(sample_memory)
    
    assert conv_id is not None
    assert conv_id == sample_memory.session_id
    
    # Check file exists
    conv_path = manager._get_conversation_path(conv_id)
    assert conv_path.exists()
    
    # Verify content
    with open(conv_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["metadata"]["conversation_id"] == conv_id
    assert data["metadata"]["turn_count"] == 2
    assert len(data["history"]) == 2


def test_save_conversation_with_custom_title(manager, sample_memory):
    """Test saving conversation with custom title."""
    custom_title = "My ML Conversation"
    conv_id = manager.save_conversation(sample_memory, title=custom_title)
    
    conversation = manager.load_conversation(conv_id)
    assert conversation is not None
    assert conversation.metadata.title == custom_title


def test_load_conversation(manager, sample_memory):
    """Test loading a saved conversation."""
    conv_id = manager.save_conversation(sample_memory)
    
    conversation = manager.load_conversation(conv_id)
    
    assert conversation is not None
    assert conversation.metadata.conversation_id == conv_id
    assert conversation.metadata.turn_count == 2
    assert len(conversation.history) == 2
    assert conversation.history[0]["query"] == "What is machine learning?"


def test_load_nonexistent_conversation(manager):
    """Test loading conversation that doesn't exist."""
    conversation = manager.load_conversation("nonexistent-id")
    assert conversation is None


def test_restore_to_memory(manager, sample_memory):
    """Test restoring conversation to memory."""
    conv_id = manager.save_conversation(sample_memory)
    
    restored_memory = manager.restore_to_memory(conv_id)
    
    assert restored_memory is not None
    assert restored_memory.get_turn_count() == 2
    assert restored_memory.session_id == conv_id
    assert restored_memory.get_last_query() == "What are neural networks?"


def test_list_conversations(manager, sample_memory):
    """Test listing conversations."""
    # Save multiple conversations
    memory1 = sample_memory
    conv_id1 = manager.save_conversation(memory1, title="Conversation 1")
    
    memory2 = ConversationMemory()
    memory2.add_turn(
        query="Test query",
        answer="Test answer",
    )
    conv_id2 = manager.save_conversation(memory2, title="Conversation 2")
    
    conversations = manager.list_conversations()
    
    assert len(conversations) == 2
    titles = [c.title for c in conversations]
    assert "Conversation 1" in titles
    assert "Conversation 2" in titles


def test_list_conversations_with_limit(manager):
    """Test listing conversations with limit."""
    # Create multiple conversations
    for i in range(10):
        memory = ConversationMemory()
        memory.add_turn(query=f"Query {i}", answer=f"Answer {i}")
        manager.save_conversation(memory, title=f"Conv {i}")
    
    # Cleanup should keep only max_conversations (5)
    conversations = manager.list_conversations()
    assert len(conversations) <= 5


def test_delete_conversation(manager, sample_memory):
    """Test deleting a conversation."""
    conv_id = manager.save_conversation(sample_memory)
    
    # Verify it exists
    assert manager.load_conversation(conv_id) is not None
    
    # Delete it
    result = manager.delete_conversation(conv_id)
    assert result is True
    
    # Verify it's gone
    assert manager.load_conversation(conv_id) is None


def test_delete_nonexistent_conversation(manager):
    """Test deleting conversation that doesn't exist."""
    result = manager.delete_conversation("nonexistent-id")
    assert result is False


def test_rename_conversation(manager, sample_memory):
    """Test renaming a conversation."""
    conv_id = manager.save_conversation(sample_memory, title="Original Title")
    
    result = manager.rename_conversation(conv_id, "New Title")
    assert result is True
    
    conversation = manager.load_conversation(conv_id)
    assert conversation.metadata.title == "New Title"


def test_export_to_markdown(manager, sample_memory):
    """Test exporting conversation to markdown."""
    conv_id = manager.save_conversation(sample_memory, title="Test Conversation")
    
    markdown = manager.export_to_markdown(conv_id)
    
    assert markdown is not None
    assert "# Test Conversation" in markdown
    assert "What is machine learning?" in markdown
    assert "Machine learning is a subset of AI" in markdown
    assert "## Turn 1" in markdown
    assert "## Turn 2" in markdown


def test_generate_title_truncation(manager):
    """Test title generation with truncation."""
    long_query = "This is a very long query " * 10
    title = manager._generate_title(long_query, max_length=50)
    
    assert len(title) <= 50
    assert title.endswith("...")


def test_save_empty_conversation(manager):
    """Test saving empty conversation."""
    empty_memory = ConversationMemory()
    conv_id = manager.save_conversation(empty_memory)
    
    assert conv_id == ""  # Should return empty string for empty conversation


def test_cleanup_old_conversations(manager):
    """Test automatic cleanup of old conversations."""
    # Create conversations exceeding max limit
    for i in range(10):
        memory = ConversationMemory()
        memory.add_turn(query=f"Query {i}", answer=f"Answer {i}")
        manager.save_conversation(memory, title=f"Conv {i}")
    
    # Check that only max_conversations remain
    conversations = manager.list_conversations()
    assert len(conversations) <= manager.max_conversations


def test_conversation_metadata():
    """Test ConversationMetadata serialization."""
    metadata = ConversationMetadata(
        conversation_id="test-id",
        title="Test",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        turn_count=5,
    )
    
    # Test to_dict and from_dict
    data = metadata.to_dict()
    restored = ConversationMetadata.from_dict(data)
    
    assert restored.conversation_id == metadata.conversation_id
    assert restored.title == metadata.title
    assert restored.turn_count == metadata.turn_count


def test_saved_conversation_serialization(sample_memory):
    """Test SavedConversation serialization."""
    metadata = ConversationMetadata(
        conversation_id="test-id",
        title="Test",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        turn_count=2,
    )
    
    conversation = SavedConversation(
        metadata=metadata,
        history=sample_memory.get_history(),
    )
    
    # Test to_dict and from_dict
    data = conversation.to_dict()
    restored = SavedConversation.from_dict(data)
    
    assert restored.metadata.conversation_id == conversation.metadata.conversation_id
    assert len(restored.history) == len(conversation.history)
