"""
Main Streamlit application for RAG System.

Chat-First UX with:
- Modern chat interface as main page
- Conversation management in sidebar
- Secondary features accessible via menu
"""

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from pathlib import Path

from config.settings import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Research Paper Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Authentication with streamlit-authenticator (Admin only)
# ============================================================================

# Load user credentials from users.yaml
users_file = Path(__file__).parent.parent / "data" / "users.yaml"
if users_file.exists():
    with open(users_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
else:
    # Fallback config if users.yaml doesn't exist
    config = {
        'credentials': {
            'usernames': {}
        },
        'cookie': {
            'expiry_days': 30,
            'key': 'rag_auth_cookie',
            'name': 'rag_auth'
        }
    }

# Create authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config.get('cookie', {}).get('name', 'rag_auth'),
    config.get('cookie', {}).get('key', 'rag_auth_cookie'),
    config.get('cookie', {}).get('expiry_days', 30),
)

# Login form
try:
    authenticator.login(location='main')
except Exception as e:
    logger.error(f"Login error: {e}")
    st.error("Login failed. Please try again.")

# Check authentication status
if st.session_state.get("authentication_status") is None:
    st.title("🔬 Research Paper Chat")
    st.info("Vui lòng đăng nhập để sử dụng hệ thống.")
    st.stop()
elif st.session_state.get("authentication_status") is False:
    st.title("🔬 Research Paper Chat")
    st.error("Sai tên đăng nhập hoặc mật khẩu!")
    st.stop()

# Custom CSS for chat-first design
st.markdown(
    """
    <style>
    .main {
        padding: 0.5rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .conversation-item {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: background 0.2s;
    }
    .conversation-item:hover {
        background-color: #f0f2f6;
    }
    .conversation-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .new-chat-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_conversation_sidebar():
    """Render conversation list in sidebar."""
    with st.sidebar:
        # User info and logout
        user_name = st.session_state.get("name", st.session_state.get("username", "Admin"))
        st.markdown(f"👤 **{user_name}**")
        authenticator.logout("🚪 Logout", location="sidebar")
        
        st.divider()
        
        # New chat button
        if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True, type="primary"):
            # Clear current conversation
            from src.core.memory import ConversationMemory
            st.session_state.conversation_memory = ConversationMemory(
                max_turns=settings.memory_max_turns,
                max_tokens=settings.memory_max_tokens,
            )
            st.session_state.chat_messages = []
            if "loaded_conversation_id" in st.session_state:
                del st.session_state.loaded_conversation_id
            st.rerun()
        
        st.divider()
        
        # Conversation list
        if settings.enable_chat_history:
            st.markdown("**💬 Conversations**")
            
            try:
                from ui.components.chat_history_manager import get_history_manager
                manager = get_history_manager()
                conversations = manager.list_conversations(limit=20)
                
                if conversations:
                    # Search box
                    search = st.text_input("🔍 Search", placeholder="Filter...", label_visibility="collapsed")
                    
                    # Filter conversations
                    if search:
                        filtered = [c for c in conversations if search.lower() in c.title.lower()]
                    else:
                        filtered = conversations
                    
                    # Display conversations
                    loaded_id = st.session_state.get("loaded_conversation_id")
                    
                    for conv in filtered:
                        is_active = conv.conversation_id == loaded_id
                        
                        # Create container for each conversation
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                if st.button(
                                    f"{'📂' if is_active else '💬'} {conv.title[:35]}",
                                    key=f"conv_{conv.conversation_id}",
                                    help=f"{conv.turn_count} messages",
                                    use_container_width=True,
                                    type="primary" if is_active else "secondary"
                                ):
                                    # Load full conversation first
                                    full_conversation = manager.load_conversation(conv.conversation_id)
                                    if full_conversation:
                                        # Restore to memory
                                        memory = manager.restore_to_memory(
                                            conv.conversation_id,
                                            max_turns=settings.memory_max_turns,
                                            max_tokens=settings.memory_max_tokens,
                                        )
                                        if memory:
                                            st.session_state.conversation_memory = memory
                                            st.session_state.loaded_conversation_id = conv.conversation_id
                                            
                                            # Rebuild chat messages from history for UI display
                                            st.session_state.chat_messages = []
                                            for turn in full_conversation.history:
                                                # Add user message
                                                st.session_state.chat_messages.append({
                                                    "role": "user",
                                                    "content": turn["query"]
                                                })
                                                # Add assistant message
                                                st.session_state.chat_messages.append({
                                                    "role": "assistant",
                                                    "content": turn["answer"],
                                                    "sources": turn.get("sources", [])
                                                })
                                            
                                            logger.info(f"Loaded conversation {conv.conversation_id} with {len(st.session_state.chat_messages)} messages")
                                            st.rerun()
                            
                            with col2:
                                if st.button("🗑️", key=f"del_{conv.conversation_id}", help="Delete"):
                                    if conv.conversation_id != loaded_id:
                                        manager.delete_conversation(conv.conversation_id)
                                        st.rerun()
                else:
                    st.info("No conversations yet")
                    
            except Exception as e:
                logger.error(f"Error rendering conversation list: {e}", exc_info=True)
                st.error("Failed to load conversations")
        
        st.divider()
        
        # System info
        st.markdown("**📊 System Info**")
        
        from src.embedding import default_vector_store
        from src.generation import default_llm_client
        
        try:
            doc_count = default_vector_store.count()
            unique_docs = len(default_vector_store.list_documents())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Papers", unique_docs)
            with col2:
                st.metric("Chunks", doc_count)
            
            usage = default_llm_client.get_usage_stats()
            st.metric("Cost", f"${usage['total_cost']:.4f}")
            
        except Exception as e:
            st.warning(f"Stats unavailable")
        
        st.divider()
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.session_state.top_k = st.slider(
                "Results", min_value=1, max_value=10, value=5
            )
            st.session_state.temperature = st.slider(
                "Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1
            )
        
        st.divider()
        
        # Secondary pages menu
        st.markdown("**📁 Tools**")
        
        if st.button("📤 Upload Papers", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
        
        if st.button("📚 Manage Library", use_container_width=True):
            st.session_state.page = "library"
            st.rerun()
        
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.page = "analytics"
            st.rerun()
        
        if st.button("💬 All Conversations", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()


def main():
    """Main application."""
    
    # Initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "chat"
    
    # Render sidebar
    render_conversation_sidebar()
    
    # Render main content based on page
    if st.session_state.page == "chat":
        from ui.components.chat_interface import render_chat_interface
        render_chat_interface()
        
    elif st.session_state.page == "upload":
        st.title("📤 Upload Papers")
        if st.button("← Back to Chat"):
            st.session_state.page = "chat"
            st.rerun()
        st.divider()
        from ui.components.document_upload import render_document_upload
        render_document_upload()
        
    elif st.session_state.page == "library":
        st.title("📚 Manage Library")
        if st.button("← Back to Chat"):
            st.session_state.page = "chat"
            st.rerun()
        st.divider()
        from ui.components.document_manager import render_document_manager
        render_document_manager()
        
    elif st.session_state.page == "analytics":
        st.title("📊 Analytics")
        if st.button("← Back to Chat"):
            st.session_state.page = "chat"
            st.rerun()
        st.divider()
        try:
            from ui.pages.analytics import render_analytics_dashboard
            render_analytics_dashboard()
        except ImportError as e:
            st.error(f"Analytics not available: {e}")
    
    elif st.session_state.page == "history":
        st.title("💬 All Conversations")
        if st.button("← Back to Chat"):
            st.session_state.page = "chat"
            st.rerun()
        st.divider()
        try:
            from ui.components.chat_history_manager import render_chat_history_manager
            render_chat_history_manager()
        except ImportError as e:
            st.error(f"Chat history not available: {e}")


if __name__ == "__main__":
    main()
