"""
Main Streamlit application for RAG System.

Phase 2 Enhanced with:
- Query interface (with hybrid search, reranking, memory)
- Document upload
- Document management & preview
- Analytics dashboard
- Export functionality
"""

import streamlit as st

from ui.components.document_manager import render_document_manager
from ui.components.document_upload import render_document_upload
from ui.components.query_interface import render_query_interface

# Page configuration
st.set_page_config(
    page_title="Scientific Paper Search",
    page_icon="📚",
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main application."""
    # Header
    st.title("📚 Scientific Paper Search System")
    st.markdown("*Quickly find relevant papers and excerpts from your research library*")

    # Sidebar navigation
    with st.sidebar:
        st.header("📁 Navigation")
        page = st.radio(
            "Select a page:",
            [
                "🔍 Search Papers",
                "📤 Upload Papers", 
                "📚 Manage Library",
                "📄 Paper Preview",
                "📊 Analytics",
            ],
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

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunks", doc_count)
            with col2:
                st.metric("Papers", unique_docs)

            # LLM stats
            usage = default_llm_client.get_usage_stats()
            st.metric("Total Cost", f"${usage['total_cost']:.4f}")

        except Exception as e:
            st.warning(f"Could not load stats: {str(e)[:50]}")

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

        st.divider()
        
        # Phase 2 features status
        with st.expander("🚀 Phase 2 Features"):
            st.markdown("**✅ Enabled:**")
            st.markdown("- 🔀 Hybrid Search (BM25+Vector)")
            st.markdown("- 📊 Cohere Reranking")
            st.markdown("- 💭 Conversation Memory")
            st.markdown("- 📈 Analytics Dashboard")
            st.markdown("- 📥 Export (MD/PDF)")
            st.markdown("- 🌐 REST API (/docs)")

    # Main content based on selected page
    if page == "🔍 Search Papers":
        render_query_interface()
        
    elif page == "📤 Upload Papers":
        render_document_upload()
        
    elif page == "📚 Manage Library":
        render_document_manager()
        
    elif page == "📄 Paper Preview":
        try:
            from ui.components.document_preview import render_document_preview
            render_document_preview()
        except ImportError as e:
            st.error(f"Document preview not available: {e}")
            
    elif page == "📊 Analytics":
        try:
            from ui.pages.analytics import render_analytics_dashboard
            render_analytics_dashboard()
        except ImportError as e:
            st.error(f"Analytics not available: {e}")
            st.info("Make sure all dependencies are installed.")

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <small>RAG System v2.0 | Phase 2 Complete | Built with ❤️ using Streamlit, OpenAI, Cohere, and Chroma</small>
            <br>
            <small>API: <a href="http://localhost:8000/docs" target="_blank">http://localhost:8000/docs</a></small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
