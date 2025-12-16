"""
Query interface component for Streamlit UI.

Enhanced with Phase 2 features:
- Hybrid search (BM25 + Vector)
- Cohere reranking
- Conversation memory (multi-turn chat)
"""

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


def render_query_interface():
    """Render query interface with Phase 2 enhancements."""
    st.header("🔍 Search Your Papers")
    st.markdown("Ask questions about your research papers and get AI-powered answers with excerpts.")

    # Check if there are documents
    try:
        doc_count = default_vector_store.count()
        if doc_count == 0:
            st.warning(
                "⚠️ No papers in the library yet. Please upload research papers first!",
                icon="📭",
            )
            return
    except Exception as e:
        st.error(f"Failed to check document count: {e}")
        return

    # Memory controls (if enabled)
    memory = get_session_memory() if settings.enable_memory else None
    
    if memory and memory.get_turn_count() > 0:
        col_mem1, col_mem2 = st.columns([4, 1])
        with col_mem1:
            st.info(f"💭 Conversation: {memory.get_turn_count()} turns", icon="💬")
        with col_mem2:
            if st.button("🗑️ Clear", help="Clear conversation history"):
                memory.clear()
                st.rerun()

    # Query input
    query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., What methods are used for text classification? What are the main findings about transformer models?",
        height=100,
        help="Ask questions about your papers. Use follow-up questions for deeper exploration.",
    )

    # Search options
    with st.expander("⚙️ Search Options", expanded=False):
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
            use_memory = st.checkbox(
                "💭 Use Conversation",
                value=settings.enable_memory and memory is not None,
                help="Include previous conversation context",
                disabled=not settings.enable_memory,
            )

    # Query type selection
    col1, col2 = st.columns([3, 1])
    with col1:
        query_type = st.radio(
            "Query Type:",
            ["Simple", "Analytical"],
            horizontal=True,
            help="Simple: Quick factual questions | Analytical: Detailed analysis",
        )
    with col2:
        search_button = st.button("🔍 Search Papers", type="primary", use_container_width=True)

    # Process query
    if search_button and query:
        process_query(
            query=query,
            query_type=query_type,
            use_hybrid=use_hybrid,
            hybrid_alpha=hybrid_alpha,
            use_rerank=use_rerank,
            use_memory=use_memory and memory is not None,
            memory=memory,
        )
    elif search_button:
        st.warning("Please enter a research question first!")

    # Query history
    if "query_history" in st.session_state and st.session_state.query_history:
        st.divider()
        with st.expander("📜 Query History"):
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                st.markdown(f"**Q{len(st.session_state.query_history) - i}:** {item['query']}")
                with st.container():
                    st.markdown(f"*{item['answer'][:200]}...*")
                st.divider()


def process_query(
    query: str,
    query_type: str,
    use_hybrid: bool = True,
    hybrid_alpha: float = 0.5,
    use_rerank: bool = True,
    use_memory: bool = True,
    memory: ConversationMemory | None = None,
):
    """Process a user query with Phase 2 enhancements."""
    # Initialize query history
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    # Get settings
    top_k = st.session_state.get("top_k", 5)

    # Search
    with st.spinner("🔍 Searching relevant documents..."):
        try:
            if use_hybrid:
                # Use hybrid retriever
                hybrid_retriever = get_hybrid_retriever()
                if hybrid_retriever:
                    results = hybrid_retriever.search(
                        query=query,
                        top_k=top_k * 2 if use_rerank else top_k,
                        alpha=hybrid_alpha,
                    )
                    search_method = "Hybrid (BM25 + Vector)"
                else:
                    # Fallback to vector search
                    results = default_vector_store.search(query, top_k=top_k * 2 if use_rerank else top_k)
                    search_method = "Vector"
            else:
                # Pure vector search
                results = default_vector_store.search(query, top_k=top_k * 2 if use_rerank else top_k)
                search_method = "Vector"

            if not results:
                st.warning("No relevant documents found for your query.")
                return

            st.info(f"📡 Search method: {search_method} | Found: {len(results)} chunks")

        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            logger.error(f"Search failed: {e}", exc_info=True)
            return

    # Reranking
    if use_rerank and results:
        with st.spinner("📊 Reranking results..."):
            try:
                reranker = get_reranker()
                if reranker:
                    results = reranker.rerank(
                        query=query,
                        documents=results,
                        top_n=top_k,
                    )
                    st.success(f"✅ Reranked to top {len(results)} results")
                else:
                    # Trim results without reranking
                    results = results[:top_k]
                    st.warning("⚠️ Reranker not available (check COHERE_API_KEY)")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                results = results[:top_k]
                st.warning(f"⚠️ Reranking skipped: {e}")
    else:
        results = results[:top_k]

    # Show retrieved documents
    with st.expander(f"📚 Papers Found ({len(results)})"):
        for i, result in enumerate(results, 1):
            # Show score based on search method
            score_info = ""
            if "rerank_score" in result:
                score_info = f"Rerank: {result['rerank_score']:.3f}"
            elif "rrf_score" in result:
                score_info = f"RRF: {result['rrf_score']:.4f}"
            elif "similarity" in result:
                score_info = f"Similarity: {result['similarity']:.3f}"
            
            st.markdown(f"**Paper {i}** ({score_info})")
            
            # Get paper title from metadata
            metadata = result.get('metadata', {})
            paper_title = metadata.get('title', '').strip() or metadata.get('filename', 'Unknown')
            st.markdown(f"*📄 {paper_title}*")
            st.text_area(
                f"Excerpt {i}",
                result.get("document", ""),
                height=100,
                key=f"result_{i}_{hash(query)}",
                label_visibility="collapsed",
            )
            st.divider()

    # Get conversation context
    conversation_context = ""
    if use_memory and memory and memory.get_turn_count() > 0:
        conversation_context = memory.get_context()
        st.info(f"💭 Using {memory.get_turn_count()} previous turns for context")

    # Generate answer
    with st.spinner("🤖 Generating AI answer..."):
        try:
            # Map query type
            query_type_map = {"Simple": "SIMPLE", "Analytical": "ANALYTICAL"}
            mapped_type = query_type_map[query_type]

            response = default_synthesizer.synthesize(
                query=query,
                retrieved_docs=results,
                query_type=mapped_type,
                conversation_context=conversation_context,
            )

            # Display answer
            st.divider()
            st.subheader("✨ Answer")

            # Use st.markdown to properly render LaTeX formulas
            # Convert \(...\) to $...$ for inline math
            answer_text = response['answer']
            # Convert LaTeX delimiters for Streamlit compatibility
            import re
            answer_text = re.sub(r'\\\((.+?)\\\)', r'$\1$', answer_text)
            answer_text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', answer_text)
            
            # Display in a styled container
            st.container(border=True)
            st.markdown(answer_text)

            # Show sources
            if response.get("sources"):
                st.markdown("**📖 Sources:**")
                for source in response["sources"]:
                    st.markdown(f"- `{source}`")

            # Show usage stats
            if "token_usage" in response:
                usage = response["token_usage"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prompt Tokens", usage["total_prompt_tokens"])
                with col2:
                    st.metric("Completion Tokens", usage["total_completion_tokens"])
                with col3:
                    st.metric("Cost", f"${usage['total_cost']:.6f}")

            # Add to conversation memory
            if memory:
                memory.add_turn(
                    query=query,
                    answer=response["answer"],
                    sources=response.get("sources", []),
                )

            # Add to history
            st.session_state.query_history.append(
                {"query": query, "answer": response["answer"], "sources": response.get("sources", [])}
            )

            st.success("✅ Answer generated successfully!")

        except Exception as e:
            st.error(f"Failed to generate answer: {str(e)}")
            logger.error(f"Answer generation failed: {e}", exc_info=True)


__all__ = ["render_query_interface"]
