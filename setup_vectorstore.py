"""
Script to initialize the vector store with product documentation.
Run this script once before starting the support agent system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.vector_store import VectorStoreManager
from utils.config import validate_config, PRODUCT_DOCS_DIR


def main():
    """Initialize the vector store."""
    print("TaskFlow Pro - Vector Store Initialization")

    try:
        validate_config()
        print("\n Configuration validated")
    except ValueError as e:
        print(f"\n Configuration error:\n{e}")
        sys.exit(1)

    if not PRODUCT_DOCS_DIR.exists():
        print(f"\nProduct documentation directory not found: {PRODUCT_DOCS_DIR}")
        sys.exit(1)

    doc_files = list(PRODUCT_DOCS_DIR.glob("**/*.md"))
    if not doc_files:
        print(f"\nNo markdown files found in {PRODUCT_DOCS_DIR}")
        sys.exit(1)

    print(f"\nFound {len(doc_files)} documentation files:")
    for doc_file in doc_files:
        print(f"  - {doc_file.relative_to(PRODUCT_DOCS_DIR)}")

    # Ask for confirmation
    print("\n" + "=" * 60)
    response = input("Initialize vector store? This may take a few minutes. (y/n): ")

    if response.lower() != "y":
        print("Cancelled.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("Starting vector store initialization...")
    print("=" * 60 + "\n")

    try:
        manager = VectorStoreManager()
        manager.initialize(force_recreate=True)

        print("Vector store initialized successfully")

        print("\nRunning test query...")
        test_query = "How do I reset my password?"
        results = manager.similarity_search(test_query, k=2)

        print(f"\nTest Query: '{test_query}'")
        print(f"Found {len(results)} relevant results:")

        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"\n{i}. {source}")
            print(f"   Preview: {preview}...")

    except Exception as e:
        print(f"\nError initializing vector store: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
