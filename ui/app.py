"""
Main Streamlit application for RAG System.

This provides a web interface for:
- Document upload
- Query interface
- Document management
- Results display
"""

import streamlit as st

from ui.components.document_manager import render_document_manager
from ui.components.document_upload import render_document_upload
from ui.components.query_interface import render_query_interface

# Page configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .query-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #4CAF50;
    }
    .citation {
        color: #0066cc;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main application."""
    # Header
    st.title("🤖 RAG System - AI-Powered Document Analysis")
    st.markdown("*Ask questions about your documents and get AI-powered answers with citations*")

    # Sidebar navigation
    with st.sidebar:
        st.header("📁 Navigation")
        page = st.radio(
            "Select a page:",
            ["💬 Query Documents", "📤 Upload Documents", "📚 Manage Documents"],
            label_visibility="collapsed",
        )

        st.divider()

        # System info
        st.header("ℹ️ System Info")

        # Import here to avoid circular imports
        from src.embedding import default_vector_store
        from src.generation import default_llm_client

        try:
            doc_count = default_vector_store.count()
            unique_docs = len(default_vector_store.list_documents())

            st.metric("Total Chunks", doc_count)
            st.metric("Documents", unique_docs)

            # LLM stats
            usage = default_llm_client.get_usage_stats()
            st.metric("Total Cost", f"${usage['total_cost']:.4f}")

        except Exception as e:
            st.warning(f"Could not load stats: {str(e)}")

        st.divider()

        # Settings
        with st.expander("⚙️ Settings"):
            st.session_state.top_k = st.slider(
                "Number of results", min_value=1, max_value=10, value=5, key="top_k_slider"
            )
            st.session_state.temperature = st.slider(
                "LLM Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                key="temp_slider",
            )

    # Main content based on selected page
    if page == "💬 Query Documents":
        render_query_interface()
    elif page == "📤 Upload Documents":
        render_document_upload()
    elif page == "📚 Manage Documents":
        render_document_manager()

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <small>RAG System MVP | Built with ❤️ using Streamlit, OpenAI, and Chroma</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
