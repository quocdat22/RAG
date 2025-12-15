"""
Document manager component for Streamlit UI.
"""

import streamlit as st
from pathlib import Path

from src.core.logging import get_logger
from src.embedding import default_vector_store

logger = get_logger(__name__)


def render_document_manager():
    """Render document management interface."""
    st.header("📚 Manage Documents")
    st.markdown("View and manage documents in your knowledge base.")

    try:
        # Get document list
        doc_ids = default_vector_store.list_documents()
        total_chunks = default_vector_store.count()

        if not doc_ids:
            st.info("📭 No documents in the system yet. Upload some documents to get started!")
            return

        # Summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", len(doc_ids))
        with col2:
            st.metric("Total Chunks", total_chunks)

        st.divider()

        # Document list
        st.subheader("📄 Documents")

        for i, doc_id in enumerate(doc_ids, 1):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    # Get first chunk to show metadata
                    try:
                        results = default_vector_store.collection.get(
                            where={"doc_id": doc_id}, limit=1, include=["metadatas"]
                        )

                        if results["metadatas"]:
                            metadata = results["metadatas"][0]
                            filename = metadata.get("filename", doc_id)
                            category = metadata.get("category", "N/A")
                            file_type = metadata.get("file_type", "N/A")

                            st.markdown(f"**{i}. {filename}**")
                            st.caption(f"Type: {file_type} | Category: {category}")
                        else:
                            st.markdown(f"**{i}. {doc_id}**")

                    except Exception as e:
                        st.markdown(f"**{i}. {doc_id}**")
                        logger.error(f"Failed to get metadata: {e}")

                with col2:
                    # Count chunks for this document
                    try:
                        doc_results = default_vector_store.collection.get(
                            where={"doc_id": doc_id}, include=[]
                        )
                        chunk_count = len(doc_results["ids"]) if doc_results["ids"] else 0
                        st.metric("Chunks", chunk_count, label_visibility="collapsed")
                    except Exception:
                        st.metric("Chunks", "?", label_visibility="collapsed")

                with col3:
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{doc_id}", type="secondary"):
                        delete_document(doc_id)

                st.divider()

        # Bulk actions
        st.subheader("⚙️ Bulk Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                confirm_clear_all()

        with col2:
            if st.button("📊 Show Statistics", type="secondary"):
                show_statistics()

    except Exception as e:
        st.error(f"Failed to load documents: {str(e)}")
        logger.error(f"Document manager error: {e}", exc_info=True)


def delete_document(doc_id: str):
    """Delete a document and its chunks."""
    try:
        with st.spinner(f"Deleting {doc_id}..."):
            deleted_count = default_vector_store.delete_by_doc_id(doc_id)

        if deleted_count > 0:
            st.success(f"✅ Deleted {deleted_count} chunks from document")
            st.rerun()  # Refresh the page
        else:
            st.warning("No chunks were deleted")

    except Exception as e:
        st.error(f"Failed to delete document: {str(e)}")
        logger.error(f"Delete failed: {e}", exc_info=True)


def confirm_clear_all():
    """Confirm and clear all documents."""
    st.warning("⚠️ This will delete ALL documents and chunks from the system!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, Delete All", type="primary", key="confirm_delete"):
            try:
                with st.spinner("Clearing all documents..."):
                    default_vector_store.reset()
                st.success("✅ All documents have been deleted!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear documents: {str(e)}")

    with col2:
        if st.button("❌ Cancel", key="cancel_delete"):
            st.info("Cancelled")


def show_statistics():
    """Show detailed statistics."""
    try:
        doc_ids = default_vector_store.list_documents()

        st.subheader("📊 Detailed Statistics")

        # Per-document stats
        stats_data = []
        for doc_id in doc_ids:
            try:
                doc_results = default_vector_store.collection.get(
                    where={"doc_id": doc_id}, include=["metadatas"]
                )

                chunk_count = len(doc_results["ids"]) if doc_results["ids"] else 0

                if doc_results["metadatas"]:
                    metadata = doc_results["metadatas"][0]
                    filename = metadata.get("filename", doc_id)
                    category = metadata.get("category", "N/A")
                    file_type = metadata.get("file_type", "N/A")

                    stats_data.append(
                        {
                            "Filename": filename,
                            "Type": file_type,
                            "Category": category,
                            "Chunks": chunk_count,
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to get stats for {doc_id}: {e}")

        if stats_data:
            import pandas as pd

            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True)

            # Summary
            st.markdown("**Summary:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(stats_data))
            with col2:
                st.metric("Total Chunks", sum(d["Chunks"] for d in stats_data))
            with col3:
                st.metric("Avg Chunks/Doc", f"{sum(d['Chunks'] for d in stats_data) / len(stats_data):.1f}")

    except Exception as e:
        st.error(f"Failed to show statistics: {str(e)}")
        logger.error(f"Statistics error: {e}", exc_info=True)


__all__ = ["render_document_manager"]
