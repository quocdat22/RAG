"""
Chat History Manager UI Component.

Provides UI for viewing, loading, deleting, and managing saved conversations.
"""

import streamlit as st
from datetime import datetime

from config.settings import settings
from src.core.chat_history import ChatHistoryManager
from src.core.logging import get_logger

logger = get_logger(__name__)


def get_history_manager() -> ChatHistoryManager:
    """Get or create chat history manager."""
    if "chat_history_manager" not in st.session_state:
        st.session_state.chat_history_manager = ChatHistoryManager(
            storage_dir=settings.conversations_dir,
            max_conversations=settings.max_saved_conversations,
        )
    return st.session_state.chat_history_manager


def render_chat_history_manager():
    """Render chat history management page."""
    st.header("💬 Chat History")
    st.markdown("View and manage your saved conversations.")
    
    if not settings.enable_chat_history:
        st.warning("Chat history is disabled in settings.")
        return
    
    manager = get_history_manager()
    
    # Get all conversations
    conversations = manager.list_conversations()
    
    if not conversations:
        st.info("📭 No saved conversations yet. Start a conversation in Search Papers to save it!")
        return
    
    # Display stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Conversations", len(conversations))
    with col2:
        total_turns = sum(c.turn_count for c in conversations)
        st.metric("Total Turns", total_turns)
    
    st.divider()
    
    # Search/filter
    search_query = st.text_input(
        "🔍 Search conversations",
        placeholder="Search by title...",
        help="Filter conversations by title",
    )
    
    # Filter conversations
    if search_query:
        filtered = [c for c in conversations if search_query.lower() in c.title.lower()]
    else:
        filtered = conversations
    
    st.markdown(f"**Showing {len(filtered)} conversation(s)**")
    
    # Display conversations
    for conv in filtered:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{conv.title}**")
                # Parse datetime
                try:
                    updated = datetime.fromisoformat(conv.updated_at)
                    time_str = updated.strftime("%Y-%m-%d %H:%M")
                except:
                    time_str = conv.updated_at
                st.caption(f"📅 {time_str} | 💬 {conv.turn_count} turns")
            
            with col2:
                if st.button("📂 Load", key=f"load_{conv.conversation_id}", help="Load this conversation"):
                    load_conversation(conv.conversation_id)
            
            with col3:
                if st.button("📥 Export", key=f"export_{conv.conversation_id}", help="Export to markdown"):
                    export_conversation(conv.conversation_id)
            
            with col4:
                if st.button("🗑️ Delete", key=f"delete_{conv.conversation_id}", help="Delete conversation"):
                    delete_conversation(conv.conversation_id)
    
    # Bulk actions
    if len(conversations) > 0:
        st.divider()
        with st.expander("⚙️ Bulk Actions"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Delete All Conversations", type="secondary"):
                    if st.session_state.get("confirm_delete_all"):
                        delete_all_conversations()
                        st.session_state.confirm_delete_all = False
                    else:
                        st.session_state.confirm_delete_all = True
                        st.warning("⚠️ Click again to confirm deletion of all conversations.")
                        st.rerun()


def load_conversation(conversation_id: str):
    """Load a conversation into current session."""
    manager = get_history_manager()
    
    try:
        # Restore to memory
        memory = manager.restore_to_memory(
            conversation_id,
            max_turns=settings.memory_max_turns,
            max_tokens=settings.memory_max_tokens,
        )
        
        if memory:
            # Update session state
            st.session_state.conversation_memory = memory
            st.session_state.loaded_conversation_id = conversation_id
            
            st.success(f"✅ Loaded conversation with {memory.get_turn_count()} turns!")
            st.info("💡 Navigate to 'Search Papers' to continue this conversation.")
        else:
            st.error("Failed to load conversation.")
            
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        logger.error(f"Failed to load conversation {conversation_id}: {e}", exc_info=True)


def export_conversation(conversation_id: str):
    """Export conversation to markdown."""
    manager = get_history_manager()
    
    try:
        markdown = manager.export_to_markdown(conversation_id)
        if markdown:
            # Get conversation metadata for filename
            conv = manager.load_conversation(conversation_id)
            if conv:
                filename = f"{conv.metadata.title[:30]}.md".replace(" ", "_")
                
                st.download_button(
                    label="📄 Download Markdown",
                    data=markdown,
                    file_name=filename,
                    mime="text/markdown",
                    key=f"download_{conversation_id}",
                )
                st.success("✅ Markdown ready for download!")
        else:
            st.error("Failed to export conversation.")
            
    except Exception as e:
        st.error(f"Error exporting conversation: {e}")
        logger.error(f"Failed to export conversation {conversation_id}: {e}", exc_info=True)


def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    manager = get_history_manager()
    
    # Check if this is the loaded conversation
    if st.session_state.get("loaded_conversation_id") == conversation_id:
        st.warning("⚠️ Cannot delete currently loaded conversation. Clear memory first.")
        return
    
    try:
        if manager.delete_conversation(conversation_id):
            st.success("✅ Conversation deleted!")
            st.rerun()
        else:
            st.error("Failed to delete conversation.")
            
    except Exception as e:
        st.error(f"Error deleting conversation: {e}")
        logger.error(f"Failed to delete conversation {conversation_id}: {e}", exc_info=True)


def delete_all_conversations():
    """Delete all conversations."""
    manager = get_history_manager()
    conversations = manager.list_conversations()
    
    deleted_count = 0
    for conv in conversations:
        if manager.delete_conversation(conv.conversation_id):
            deleted_count += 1
    
    st.success(f"✅ Deleted {deleted_count} conversation(s)!")
    st.rerun()


def render_sidebar_history_widget():
    """Render compact history widget for sidebar."""
    if not settings.enable_chat_history:
        return
    
    manager = get_history_manager()
    conversations = manager.list_conversations(limit=5)
    
    if not conversations:
        return
    
    st.markdown("**💬 Recent Conversations**")
    
    # Show loaded conversation indicator
    loaded_id = st.session_state.get("loaded_conversation_id")
    if loaded_id:
        loaded_conv = manager.load_conversation(loaded_id)
        if loaded_conv:
            st.info(f"📂 Loaded: {loaded_conv.metadata.title[:30]}")
    
    # Quick load dropdown
    conv_options = {f"{c.title[:40]} ({c.turn_count} turns)": c.conversation_id for c in conversations}
    
    selected = st.selectbox(
        "Quick load:",
        options=[""] + list(conv_options.keys()),
        key="sidebar_conv_select",
        label_visibility="collapsed",
    )
    
    if selected and selected != "":
        conversation_id = conv_options[selected]
        if st.button("📂 Load Selected", key="sidebar_load_btn", use_container_width=True):
            load_conversation(conversation_id)
            st.rerun()


__all__ = ["render_chat_history_manager", "render_sidebar_history_widget", "get_history_manager"]
