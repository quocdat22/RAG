"""
Document upload component for Streamlit UI.
"""

import streamlit as st
from pathlib import Path

from src.core.logging import get_logger
from src.embedding import default_vector_store
from src.ingestion import (
    DocumentLoaderFactory,
    chunk_document,
    enrich_document_metadata,
)

logger = get_logger(__name__)


def render_document_upload():
    """Render document upload interface."""
    st.header("📤 Upload Documents")
    st.markdown("Upload documents to add them to the knowledge base for querying.")

    # Upload area
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "txt", "csv", "docx", "xlsx"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, CSV, DOCX, XLSX",
    )

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) selected")

        # Upload button
        if st.button("🚀 Process and Index Documents", type="primary"):
            process_uploaded_files(uploaded_files)


def process_uploaded_files(uploaded_files):
    """Process and index uploaded files."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_files = len(uploaded_files)
    all_chunks = []

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {i+1}/{total_files}: {uploaded_file.name}")

            # Save file temporarily
            temp_path = Path("data/documents") / uploaded_file.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load document
            with st.spinner(f"Loading {uploaded_file.name}..."):
                doc = DocumentLoaderFactory.load_document(temp_path)
                st.success(f"✅ Loaded: {uploaded_file.name}")

            # Enrich metadata
            with st.spinner(f"Enriching metadata for {uploaded_file.name}..."):
                doc = enrich_document_metadata(doc)

                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Language", doc.metadata.get("language", "N/A"))
                with col2:
                    st.metric("Category", doc.metadata.get("category", "N/A"))
                with col3:
                    st.metric("Words", doc.metadata.get("word_count", 0))

            # Chunk document
            with st.spinner(f"Chunking {uploaded_file.name}..."):
                chunks = chunk_document(doc)
                st.info(f"📦 Created {len(chunks)} chunks")
                all_chunks.extend(chunks)

            progress_bar.progress((i + 1) / total_files)

        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
            logger.error(f"Upload processing failed: {e}", exc_info=True)

    # Index all chunks
    if all_chunks:
        status_text.text(f"Indexing {len(all_chunks)} chunks in vector store...")

        try:
            with st.spinner("Indexing..."):
                indexed_count = default_vector_store.index_chunks_batch(all_chunks)

            progress_bar.progress(1.0)
            status_text.empty()

            # Success message
            st.success(f"🎉 Successfully indexed {indexed_count} chunks from {total_files} documents!")

            # Show updated stats
            total_docs = default_vector_store.count()
            unique_docs = len(default_vector_store.list_documents())

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks in System", total_docs)
            with col2:
                st.metric("Unique Documents", unique_docs)

        except Exception as e:
            st.error(f"❌ Indexing failed: {str(e)}")
            logger.error(f"Indexing failed: {e}", exc_info=True)
    else:
        status_text.empty()
        st.warning("No documents were processed successfully.")


__all__ = ["render_document_upload"]
