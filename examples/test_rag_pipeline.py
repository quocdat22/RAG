"""
Complete end-to-end RAG pipeline test.

This demonstrates the full RAG workflow:
1. Load documents
2. Chunk and enrich
3. Generate embeddings and index
4. Query with similarity search
5. Generate LLM response with citations
"""

import os
from pathlib import Path

from src.core.logging import setup_logging
from src.embedding import default_embedder, default_vector_store
from src.generation import default_llm_client, default_synthesizer
from src.ingestion import DocumentLoaderFactory, chunk_document, enrich_document_metadata


def create_sample_documents():
    """Create sample documents for testing."""
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Sample document 1: AI Overview
    doc1_path = docs_dir / "ai_overview.txt"
    if not doc1_path.exists():
        doc1_content = """
Artificial Intelligence: An Overview

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines
that are programmed to think and learn like humans. The field of AI research was founded
in 1956 at a conference at Dartmouth College.

Machine Learning
Machine learning is a subset of AI that enables systems to learn and improve from experience
without being explicitly programmed. It focuses on developing computer programs that can
access data and use it to learn for themselves.

Deep Learning
Deep learning is part of a broader family of machine learning methods based on artificial
neural networks. It has been applied to fields including computer vision, speech recognition,
natural language processing, and bioinformatics.

Applications
AI is being used in various domains:
- Healthcare: Disease diagnosis, drug discovery
- Finance: Fraud detection, algorithmic trading
- Transportation: Self-driving cars
- Customer Service: Chatbots and virtual assistants
"""
        doc1_path.write_text(doc1_content, encoding="utf-8")
        print(f"✅ Created: {doc1_path.name}")

    # Sample document 2: Python Programming
    doc2_path = docs_dir / "python_basics.txt"
    if not doc2_path.exists():
        doc2_content = """
Python Programming Basics

Python is a high-level, interpreted programming language known for its simple syntax
and readability. Created by Guido van Rossum and first released in 1991, Python emphasizes
code readability with its notable use of significant indentation.

Key Features
- Easy to learn and use
- Extensive standard library
- Dynamic typing
- Cross-platform compatibility
- Large community support

Popular Libraries
Python has a rich ecosystem of libraries:
- NumPy and Pandas for data analysis
- TensorFlow and PyTorch for machine learning
- Django and Flask for web development
- Matplotlib and Seaborn for visualization

Use Cases
Python is widely used for:
- Web development
- Data science and analytics
- Machine learning and AI
- Automation and scripting
- Scientific computing
"""
        doc2_path.write_text(doc2_content, encoding="utf-8")
        print(f"✅ Created: {doc2_path.name}")

    return [doc1_path, doc2_path]


def main():
    """Main RAG pipeline demonstration."""
    # Setup
    setup_logging(log_level="INFO")

    print("=" * 70)
    print("🚀 RAG System - End-to-End Pipeline Test")
    print("=" * 70)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set in environment")
        print("   This test will fail at LLM generation step.")
        print("   Set it with: export OPENAI_API_KEY='your-key'\n")

    # Step 1: Create and load documents
    print("\n📄 Step 1: Loading documents...")
    doc_paths = create_sample_documents()

    all_chunks = []
    for doc_path in doc_paths:
        print(f"   Processing: {doc_path.name}")

        # Load
        doc = DocumentLoaderFactory.load_document(doc_path)

        # Enrich metadata
        doc = enrich_document_metadata(doc)
        print(
            f"      Language: {doc.metadata.get('language')}, "
            f"Category: {doc.metadata.get('category')}"
        )

        # Chunk
        chunks = chunk_document(doc)
        print(f"      Created {len(chunks)} chunks")

        all_chunks.extend(chunks)

    print(f"\n   ✅ Total chunks to index: {len(all_chunks)}")

    # Step 2: Index in vector store
    print("\n🔍 Step 2: Indexing in vector store...")
    try:
        indexed_count = default_vector_store.index_chunks_batch(all_chunks)
        print(f"   ✅ Indexed {indexed_count} chunks")

        total_docs = default_vector_store.count()
        print(f"   ✅ Total documents in store: {total_docs}")

    except Exception as e:
        print(f"   ❌ Indexing failed: {e}")
        return

    # Step 3: Test queries
    print("\n💬 Step 3: Testing RAG queries...")

    test_queries = [
        "What is machine learning?",
        "Tell me about Python libraries for data science",
        "How is AI used in healthcare?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Query {i}: {query}")
        print("=" * 70)

        # Search for relevant documents
        print("\n   🔍 Searching...")
        try:
            results = default_vector_store.search(query, top_k=3)

            if not results:
                print("   ❌ No results found")
                continue

            print(f"   ✅ Found {len(results)} relevant documents:\n")

            for j, result in enumerate(results, 1):
                print(f"   Result {j}:")
                print(f"   - Similarity: {result['similarity']:.4f}")
                print(f"   - Source: {result['metadata'].get('filename', 'Unknown')}")
                print(f"   - Preview: {result['document'][:100]}...")
                print()

        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            continue

        # Generate response with LLM
        print("   🤖 Generating LLM response...")
        try:
            response = default_synthesizer.synthesize(
                query=query,
                retrieved_docs=results,
                query_type="SIMPLE",
            )

            print("\n   ✅ Generated Response:")
            print("   " + "-" * 66)
            # Print answer with indentation
            for line in response["answer"].split("\n"):
                print(f"   {line}")
            print("   " + "-" * 66)

            if response.get("sources"):
                print(f"\n   📚 Sources cited: {', '.join(response['sources'])}")

            # Show token usage
            if "token_usage" in response:
                usage = response["token_usage"]
                print(f"\n   📊 Token usage:")
                print(f"      - Prompt: {usage['total_prompt_tokens']}")
                print(f"      - Completion: {usage['total_completion_tokens']}")
                print(f"      - Total cost: ${usage['total_cost']:.6f}")

        except Exception as e:
            print(f"   ❌ LLM generation failed: {e}")
            import traceback

            traceback.print_exc()

    # Step 4: Statistics
    print(f"\n{'=' * 70}")
    print("📊 Final Statistics")
    print("=" * 70)

    print(f"   Total chunks indexed: {default_vector_store.count()}")
    print(f"   Unique documents: {len(default_vector_store.list_documents())}")

    usage = default_llm_client.get_usage_stats()
    print(f"\n   LLM Usage:")
    print(f"   - Total tokens: {usage['total_tokens']}")
    print(f"   - Total cost: ${usage['total_cost']:.6f}")
    print(f"   - Model: {usage['model']}")

    print("\n" + "=" * 70)
    print("✅ RAG Pipeline Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
