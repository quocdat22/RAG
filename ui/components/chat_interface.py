"""
Modern chat interface component for RAG system.

Uses Streamlit's native chat components for a clean ChatGPT-like experience.
"""

import re
import streamlit as st

from config.settings import settings
from src.core.logging import get_logger
from src.core.memory import ConversationMemory
from src.embedding import default_vector_store
from src.generation import default_synthesizer

logger = get_logger(__name__)


def get_session_memory() -> ConversationMemory:
    """Get or create conversation memory for current session."""
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory(
            max_turns=settings.memory_max_turns,
            max_tokens=settings.memory_max_tokens,
        )
    return st.session_state.conversation_memory


def get_hybrid_retriever():
    """Get or create hybrid retriever."""
    if "hybrid_retriever" not in st.session_state:
        try:
            from src.retrieval.hybrid_retriever import HybridRetriever
            st.session_state.hybrid_retriever = HybridRetriever()
        except Exception as e:
            logger.warning(f"Failed to initialize hybrid retriever: {e}")
            st.session_state.hybrid_retriever = None
    return st.session_state.hybrid_retriever


def get_reranker():
    """Get or create reranker."""
    if "reranker" not in st.session_state:
        try:
            from src.retrieval.reranker import CohereReranker
            reranker = CohereReranker()
            if reranker.is_available():
                st.session_state.reranker = reranker
            else:
                st.session_state.reranker = None
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
            st.session_state.reranker = None
    return st.session_state.reranker


def render_chat_interface():
    """Render modern chat interface using Streamlit's native chat components."""
    
    # Check if there are documents
    try:
        doc_count = default_vector_store.count()
        if doc_count == 0:
            st.warning(
                "⚠️ No papers in the library yet. Please upload research papers first!",
                icon="📭",
            )
            if st.button("📤 Go to Upload"):
                st.session_state.page = "upload"
                st.rerun()
            return
    except Exception as e:
        st.error(f"Failed to check document count: {e}")
        return
    
    # Initialize chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Get memory
    memory = get_session_memory() if settings.enable_memory else None
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("💬 Chat with Your Research Papers")
        if memory and memory.get_turn_count() > 0:
            st.caption(f"📝 {memory.get_turn_count()} messages in this conversation")
    with col2:
        if st.button("🗑️ Clear Chat", help="Start a new conversation"):
            if memory:
                memory.clear()
            st.session_state.chat_messages = []
            if "loaded_conversation_id" in st.session_state:
                del st.session_state.loaded_conversation_id
            st.rerun()
    
    # Chat settings in expander (above messages)
    with st.expander("⚙️ Chat Settings"):
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            use_hybrid = st.checkbox(
                "🔀 Hybrid Search",
                value=settings.enable_hybrid_search,
                help="Combine keyword (BM25) and semantic (vector) search",
            )
            hybrid_alpha = st.slider(
                "Vector ↔ Keyword",
                min_value=0.0,
                max_value=1.0,
                value=settings.hybrid_alpha,
                step=0.1,
                help="1.0 = Pure vector, 0.0 = Pure keyword",
                disabled=not use_hybrid,
            )
        
        with col_opt2:
            use_rerank = st.checkbox(
                "📊 Reranking",
                value=settings.enable_reranking,
                help="Use Cohere to rerank results for better relevance",
            )
            query_type = st.selectbox(
                "Response Style",
                ["Simple", "Analytical"],
                index=0,  # Default to Simple (cost-effective)
                help="Simple: Fast, concise (gpt-4o-mini) | Analytical: Deep reasoning (GPT-5)",
            )
            
            # Model indicator
            if query_type == "Analytical":
                st.caption("🧠 Using GPT-5 for advanced analysis")
            else:
                st.caption("⚡ Using gpt-4o-mini for quick responses")
    
    st.divider()
    
    # Display chat messages using Streamlit's native chat components
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            # Display content - Streamlit's markdown supports LaTeX natively
            st.markdown(msg["content"])
            
            # Show sources for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                st.caption("📚 **Sources:** " + ", ".join([f"`{s}`" for s in msg["sources"]]))
    
    # Chat input at bottom using Streamlit's native chat input
    if prompt := st.chat_input("Ask a question about your research papers..."):
        # Add user message to chat
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = process_chat_message(
                    user_input=prompt,
                    query_type=query_type,
                    use_hybrid=use_hybrid,
                    hybrid_alpha=hybrid_alpha,
                    use_rerank=use_rerank,
                    memory=memory,
                )
                
                if response_data:
                    # Display answer - Streamlit handles LaTeX natively
                    st.markdown(response_data["answer"])
                    
                    # Show sources
                    if response_data.get("sources"):
                        st.caption("📚 **Sources:** " + ", ".join([f"`{s}`" for s in response_data["sources"]]))
        
        # Rerun to update chat display
        st.rerun()


def process_chat_message(
    user_input: str,
    query_type: str,
    use_hybrid: bool,
    hybrid_alpha: float,
    use_rerank: bool,
    memory: ConversationMemory | None,
) -> dict | None:
    """Process a chat message and generate response."""
    
    # Get settings
    top_k = st.session_state.get("top_k", 5)
    
    try:
        # Search
        if use_hybrid:
            hybrid_retriever = get_hybrid_retriever()
            if hybrid_retriever:
                results = hybrid_retriever.search(
                    query=user_input,
                    top_k=top_k * 2 if use_rerank else top_k,
                    alpha=hybrid_alpha,
                )
            else:
                results = default_vector_store.search(user_input, top_k=top_k * 2 if use_rerank else top_k)
        else:
            results = default_vector_store.search(user_input, top_k=top_k * 2 if use_rerank else top_k)
        
        if not results:
            response_data = {
                "answer": "I couldn't find any relevant information in your papers to answer that question.",
                "sources": []
            }
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response_data["answer"],
                "sources": response_data["sources"]
            })
            return response_data
        
        # Reranking
        if use_rerank and results:
            try:
                reranker = get_reranker()
                if reranker:
                    results = reranker.rerank(
                        query=user_input,
                        documents=results,
                        top_n=top_k,
                    )
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                results = results[:top_k]
        else:
            results = results[:top_k]
        
        # Get conversation context
        conversation_context = ""
        if memory and memory.get_turn_count() > 0:
            conversation_context = memory.get_context()
        
        # Generate answer
        query_type_map = {"Simple": "SIMPLE", "Analytical": "ANALYTICAL"}
        mapped_type = query_type_map[query_type]
        
        response = default_synthesizer.synthesize(
            query=user_input,
            retrieved_docs=results,
            query_type=mapped_type,
            conversation_context=conversation_context,
        )
        
        # Add assistant message to chat
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", [])
        })
        
        # Add to memory
        if memory:
            memory.add_turn(
                query=user_input,
                answer=response["answer"],
                sources=response.get("sources", []),
            )
            
            # Auto-save conversation
            if settings.enable_chat_history and settings.auto_save_conversations:
                try:
                    from ui.components.chat_history_manager import get_history_manager
                    manager = get_history_manager()
                    
                    conversation_id = st.session_state.get("loaded_conversation_id")
                    saved_id = manager.save_conversation(
                        memory,
                        conversation_id=conversation_id,
                    )
                    st.session_state.loaded_conversation_id = saved_id
                    logger.debug(f"Auto-saved conversation {saved_id}")
                except Exception as e:
                    logger.warning(f"Failed to auto-save conversation: {e}")
        
        return response
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": error_msg,
            "sources": []
        })
        logger.error(f"Chat processing failed: {e}", exc_info=True)
        return {"answer": error_msg, "sources": []}


__all__ = ["render_chat_interface"]
