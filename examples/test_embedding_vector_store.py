"""
Example script to test embedding generation and vector store.

This script demonstrates:
1. Loading a document
2. Chunking the document
3. Generating embeddings
4. Indexing in vector store
5. Performing similarity search
"""

from pathlib import Path

from src.core.logging import setup_logging
from src.embedding import default_embedder, default_vector_store
from src.ingestion import DocumentLoaderFactory, chunk_document, enrich_document_metadata


def main():
    """Main function to test embedding and vector store."""
    # Setup logging
    setup_logging(log_level="INFO")

    print("=" * 60)
    print("🚀 RAG System - Embedding & Vector Store Test")
    print("=" * 60)

    # Step 1: Test embedding generation
    print("\n📊 Step 1: Testing embedding generation...")
    test_text = "This is a test document about artificial intelligence and machine learning."

    try:
        embedding = default_embedder.generate_embedding(test_text)
        print(f"✅ Generated embedding with dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")
        return

    # Step 2: Test batch embeddings
    print("\n📊 Step 2: Testing batch embedding generation...")
    test_texts = [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "Natural language processing enables computers to understand text.",
    ]

    try:
        embeddings = default_embedder.generate_embeddings_batch(test_texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            print(f"   Text {i+1}: dimension {len(emb)}")
    except Exception as e:
        print(f"❌ Failed to generate batch embeddings: {e}")
        return

    # Step 3: Check vector store connection
    print("\n📊 Step 3: Checking vector store connection...")
    try:
        count = default_vector_store.count()
        print(f"✅ Connected to vector store")
        print(f"   Current document count: {count}")
    except Exception as e:
        print(f"❌ Failed to connect to vector store: {e}")
        return

    # Step 4: Load and process a test document (if exists)
    print("\n📊 Step 4: Testing document processing pipeline...")

    # Create a test document
    test_doc_path = Path("data/documents/test_sample.txt")
    test_doc_path.parent.mkdir(parents=True, exist_ok=True)

    if not test_doc_path.exists():
        print(f"   Creating test document: {test_doc_path}")
        test_content = """
        Artificial Intelligence and Machine Learning

        Artificial Intelligence (AI) is revolutionizing how we interact with technology.
        Machine learning, a subset of AI, enables computers to learn from data without
        being explicitly programmed.

        Deep Learning

        Deep learning is an advanced form of machine learning that uses neural networks
        with multiple layers. It has achieved remarkable success in areas like computer
        vision, natural language processing, and speech recognition.

        Applications

        AI and ML are being applied in various domains:
        - Healthcare: Disease diagnosis and drug discovery
        - Finance: Fraud detection and algorithmic trading
        - Transportation: Autonomous vehicles
        - Customer Service: Chatbots and virtual assistants

        Future Outlook

        The future of AI holds immense potential. As computational power increases and
        algorithms improve, we can expect even more sophisticated AI applications that
        will transform industries and society.
        """
        test_doc_path.write_text(test_content, encoding="utf-8")

    try:
        # Load document
        print(f"   Loading document: {test_doc_path.name}")
        document = DocumentLoaderFactory.load_document(test_doc_path)
        print(f"   ✅ Loaded document (length: {len(document.content)} chars)")

        # Enrich metadata
        document = enrich_document_metadata(document)
        print(f"   ✅ Enriched metadata:")
        print(f"      - Language: {document.metadata.get('language')}")
        print(f"      - Category: {document.metadata.get('category')}")
        print(f"      - Keywords: {', '.join(document.metadata.get('keywords', [])[:5])}")

        # Chunk document
        chunks = chunk_document(document)
        print(f"   ✅ Created {len(chunks)} chunks")

        # Index chunks in vector store
        print(f"\n   Indexing {len(chunks)} chunks...")
        indexed_count = default_vector_store.index_chunks_batch(chunks)
        print(f"   ✅ Indexed {indexed_count} chunks")

    except Exception as e:
        print(f"   ❌ Failed to process document: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 5: Test similarity search
    print("\n📊 Step 5: Testing similarity search...")
    test_queries = [
        "What is machine learning?",
        "Tell me about deep learning applications",
        "How is AI used in healthcare?",
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            results = default_vector_store.search(query, top_k=3)

            if results:
                print(f"   Found {len(results)} results:")
                for i, result in enumerate(results):
                    print(f"\n   Result {i+1}:")
                    print(f"   - Similarity: {result['similarity']:.4f}")
                    print(f"   - Text: {result['document'][:100]}...")
                    print(f"   - Source: {result['metadata'].get('filename', 'Unknown')}")
            else:
                print("   No results found")

        except Exception as e:
            print(f"   ❌ Search failed: {e}")

    # Step 6: Vector store statistics
    print("\n📊 Step 6: Vector store statistics...")
    try:
        total_chunks = default_vector_store.count()
        doc_ids = default_vector_store.list_documents()

        print(f"   Total chunks: {total_chunks}")
        print(f"   Unique documents: {len(doc_ids)}")
        if doc_ids:
            print(f"   Document IDs: {', '.join(doc_ids[:5])}")

    except Exception as e:
        print(f"   ❌ Failed to get statistics: {e}")

    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
