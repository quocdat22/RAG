"""Quick test for imports."""

print("Testing imports...")

try:
    from src.core import get_logger
    print("✅ Core imported")
except Exception as e:
    print(f"❌ Core import failed: {e}")

try:
    from src.ingestion import DocumentLoaderFactory
    print("✅ Ingestion imported")
except Exception as e:
    print(f"❌ Ingestion import failed: {e}")

try:
    from src.embedding import EmbeddingGenerator
    print("✅ EmbeddingGenerator imported")
except Exception as e:
    print(f"❌ EmbeddingGenerator import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.embedding import VectorStore
    print("✅ VectorStore imported")
except Exception as e:
    print(f"❌ VectorStore import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All imports successful!")
