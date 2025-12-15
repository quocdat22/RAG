"""
Document preview component for Streamlit UI.

Provides document preview functionality with:
- Full document display
- Chunk highlighting
- Metadata display
- Export options
"""

import streamlit as st

from src.core.logging import get_logger
from src.embedding import default_vector_store

logger = get_logger(__name__)


def render_document_preview(doc_id: str | None = None):
    """
    Render document preview modal/section.
    
    Args:
        doc_id: Document ID to preview (optional, shows selector if None)
    """
    st.subheader("📄 Document Preview")
    
    # Get available documents
    try:
        documents = default_vector_store.list_documents()
        
        if not documents:
            st.info("No documents available for preview.")
            return
            
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
        return
    
    # Document selector
    if doc_id is None:
        selected_doc = st.selectbox(
            "Select a document to preview:",
            options=documents,
            format_func=lambda x: x[:50] + "..." if len(x) > 50 else x,
        )
    else:
        selected_doc = doc_id
    
    if selected_doc:
        show_document_details(selected_doc)


def show_document_details(doc_id: str, highlight_text: str = ""):
    """
    Show detailed document information.
    
    Args:
        doc_id: Document ID
        highlight_text: Optional text to highlight in the document
    """
    try:
        # Get document chunks
        collection = default_vector_store.collection
        result = collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        
        if not result or not result.get("ids"):
            st.warning(f"Document '{doc_id}' not found.")
            return
        
        chunks = []
        for i, chunk_id in enumerate(result["ids"]):
            chunks.append({
                "id": chunk_id,
                "content": result["documents"][i] if result.get("documents") else "",
                "metadata": result["metadatas"][i] if result.get("metadatas") else {},
            })
        
        # Document info
        if chunks:
            metadata = chunks[0]["metadata"]
            
            # Header
            st.markdown(f"### 📋 {metadata.get('filename', 'Unknown Document')}")
            
            # Metadata grid
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Chunks", len(chunks))
            with col2:
                st.metric("Category", metadata.get("category", "N/A"))
            with col3:
                file_type = metadata.get("file_type", "Unknown")
                st.metric("File Type", file_type.upper())
            
            # Additional metadata
            with st.expander("📊 Metadata Details"):
                display_metadata = {
                    "Document ID": doc_id,
                    "Filename": metadata.get("filename", "Unknown"),
                    "File Type": metadata.get("file_type", "Unknown"),
                    "Category": metadata.get("category", "N/A"),
                    "Language": metadata.get("language", "N/A"),
                    "Word Count": metadata.get("word_count", "N/A"),
                    "Keywords": metadata.get("keywords", "N/A"),
                }
                
                for key, value in display_metadata.items():
                    st.markdown(f"**{key}:** {value}")
            
            st.divider()
            
            # Chunk content
            st.markdown("### 📝 Document Content")
            
            # View mode selection
            view_mode = st.radio(
                "View mode:",
                ["All Chunks", "Single Chunk", "Combined"],
                horizontal=True,
            )
            
            if view_mode == "Combined":
                # Show all chunks combined
                combined_content = "\n\n---\n\n".join([
                    chunk["content"] for chunk in chunks
                ])
                
                if highlight_text:
                    combined_content = highlight_content(combined_content, highlight_text)
                    st.markdown(combined_content, unsafe_allow_html=True)
                else:
                    st.text_area(
                        "Full Document",
                        combined_content,
                        height=400,
                        label_visibility="collapsed",
                    )
                    
            elif view_mode == "Single Chunk":
                # Chunk selector
                chunk_idx = st.number_input(
                    "Chunk number:",
                    min_value=1,
                    max_value=len(chunks),
                    value=1,
                ) - 1
                
                chunk = chunks[chunk_idx]
                
                st.markdown(f"**Chunk {chunk_idx + 1} of {len(chunks)}**")
                st.markdown(f"*ID: {chunk['id']}*")
                
                content = chunk["content"]
                if highlight_text:
                    content = highlight_content(content, highlight_text)
                    st.markdown(content, unsafe_allow_html=True)
                else:
                    st.text_area(
                        f"Chunk {chunk_idx + 1}",
                        content,
                        height=200,
                        label_visibility="collapsed",
                    )
                    
            else:  # All Chunks
                for i, chunk in enumerate(chunks):
                    with st.expander(f"Chunk {i + 1}", expanded=(i == 0)):
                        st.markdown(f"*ID: {chunk['id']}*")
                        
                        content = chunk["content"]
                        if highlight_text:
                            content = highlight_content(content, highlight_text)
                            st.markdown(content, unsafe_allow_html=True)
                        else:
                            st.text_area(
                                f"Content {i + 1}",
                                content,
                                height=150,
                                key=f"chunk_preview_{i}",
                                label_visibility="collapsed",
                            )
            
            # Export section
            st.divider()
            render_export_options(chunks, metadata)
    
    except Exception as e:
        logger.error(f"Failed to show document details: {e}")
        st.error(f"Failed to load document: {e}")


def highlight_content(content: str, highlight_text: str) -> str:
    """
    Highlight text in content.
    
    Args:
        content: Original content
        highlight_text: Text to highlight
        
    Returns:
        HTML with highlighted text
    """
    import re
    
    if not highlight_text:
        return content
    
    # Escape HTML
    content = content.replace("<", "&lt;").replace(">", "&gt;")
    
    # Highlight matches (case insensitive)
    pattern = re.compile(re.escape(highlight_text), re.IGNORECASE)
    highlighted = pattern.sub(
        lambda m: f'<mark style="background-color: #ffeb3b; padding: 2px 4px;">{m.group()}</mark>',
        content
    )
    
    return f"<div style='white-space: pre-wrap;'>{highlighted}</div>"


def render_export_options(chunks: list, metadata: dict):
    """Render export options for the document."""
    st.markdown("### 📥 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Export as Markdown", use_container_width=True):
            # Generate markdown
            md_content = generate_document_markdown(chunks, metadata)
            
            st.download_button(
                label="⬇️ Download Markdown",
                data=md_content,
                file_name=f"{metadata.get('filename', 'document')}.md",
                mime="text/markdown",
            )
    
    with col2:
        if st.button("📄 Export as Text", use_container_width=True):
            # Generate plain text
            text_content = "\n\n---\n\n".join([chunk["content"] for chunk in chunks])
            
            st.download_button(
                label="⬇️ Download Text",
                data=text_content,
                file_name=f"{metadata.get('filename', 'document')}.txt",
                mime="text/plain",
            )


def generate_document_markdown(chunks: list, metadata: dict) -> str:
    """Generate markdown representation of document."""
    parts = [
        f"# {metadata.get('filename', 'Document')}",
        "",
        "## Metadata",
        f"- **Category:** {metadata.get('category', 'N/A')}",
        f"- **Language:** {metadata.get('language', 'N/A')}",
        f"- **Chunks:** {len(chunks)}",
        "",
        "---",
        "",
        "## Content",
        "",
    ]
    
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"### Chunk {i}")
        parts.append("")
        parts.append(chunk["content"])
        parts.append("")
        parts.append("---")
        parts.append("")
    
    return "\n".join(parts)


__all__ = [
    "render_document_preview",
    "show_document_details",
    "highlight_content",
]
