"""
Query interface component for Streamlit UI.
"""

import streamlit as st

from src.core.logging import get_logger
from src.embedding import default_vector_store
from src.generation import default_synthesizer

logger = get_logger(__name__)


def render_query_interface():
    """Render query interface."""
    st.header("💬 Query Your Documents")
    st.markdown("Ask questions about your uploaded documents and get AI-powered answers.")

    # Check if there are documents
    try:
        doc_count = default_vector_store.count()
        if doc_count == 0:
            st.warning(
                "⚠️ No documents in the system yet. Please upload documents first!",
                icon="📭",
            )
            return
    except Exception as e:
        st.error(f"Failed to check document count: {e}")
        return

    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is machine learning?",
        height=100,
        help="Ask any question about your documents",
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
        search_button = st.button("🔍 Search & Answer", type="primary", use_container_width=True)

    # Process query
    if search_button and query:
        process_query(query, query_type)
    elif search_button:
        st.warning("Please enter a question first!")

    # Query history
    if "query_history" in st.session_state and st.session_state.query_history:
        st.divider()
        with st.expander("📜 Query History"):
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                st.markdown(f"**Q{len(st.session_state.query_history) - i}:** {item['query']}")
                with st.container():
                    st.markdown(f"*{item['answer'][:200]}...*")
                st.divider()


def process_query(query: str, query_type: str):
    """Process a user query."""
    # Initialize query history
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    # Get settings
    top_k = st.session_state.get("top_k", 5)
    temperature = st.session_state.get("temperature", 0.1)

    # Search
    with st.spinner("🔍 Searching relevant documents..."):
        try:
            results = default_vector_store.search(query, top_k=top_k)

            if not results:
                st.warning("No relevant documents found for your query.")
                return

            st.success(f"✅ Found {len(results)} relevant chunks")

        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            logger.error(f"Search failed: {e}", exc_info=True)
            return

    # Show retrieved documents
    with st.expander(f"📚 Retrieved Documents ({len(results)})"):
        for i, result in enumerate(results, 1):
            st.markdown(f"**Result {i}** (Similarity: {result['similarity']:.3f})")
            st.markdown(f"*Source: {result['metadata'].get('filename', 'Unknown')}*")
            st.text_area(
                f"Content {i}",
                result["document"],
                height=100,
                key=f"result_{i}",
                label_visibility="collapsed",
            )
            st.divider()

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
            )

            # Display answer
            st.divider()
            st.subheader("✨ Answer")

            st.markdown(
                f"""
                <div style='background-color: #f0f9ff; padding: 1.5rem; border-radius: 0.5rem; 
                            border-left: 4px solid #0066cc;'>
                    {response['answer']}
                </div>
                """,
                unsafe_allow_html=True,
            )

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

            # Add to history
            st.session_state.query_history.append(
                {"query": query, "answer": response["answer"], "sources": response.get("sources", [])}
            )

            st.success("✅ Answer generated successfully!")

        except Exception as e:
            st.error(f"Failed to generate answer: {str(e)}")
            logger.error(f"Answer generation failed: {e}", exc_info=True)


__all__ = ["render_query_interface"]
