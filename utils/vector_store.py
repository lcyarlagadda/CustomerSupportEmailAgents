"""Vector store management for RAG system."""

import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from utils.config import (
    PRODUCT_DOCS_DIR,
    VECTOR_DB_DIR,
    VECTOR_DB_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    EMBEDDING_MODEL,
)


class VectorStoreManager:
    """Manages the vector store for product documentation."""

    def __init__(self):
        """Initialize the vector store manager."""
        # Use local embeddings - no token needed for public models
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=None,
        )
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_documents(self) -> List[Document]:
        """
        Load all markdown documents from the product docs directory.

        Returns:
            List of Document objects
        """
        print(f"Loading documents from {PRODUCT_DOCS_DIR}...")

        loader = DirectoryLoader(
            str(PRODUCT_DOCS_DIR),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )

        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for better retrieval.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        print(f"Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        return chunks

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.

        Args:
            documents: List of document chunks

        Returns:
            Chroma vector store
        """
        print(f"Creating vector store at {VECTOR_DB_DIR}/{VECTOR_DB_NAME}...")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(VECTOR_DB_DIR / VECTOR_DB_NAME),
            collection_name=VECTOR_DB_NAME,
        )

        print(f"Vector store created successfully")
        return vectorstore

    def load_vectorstore(self) -> Chroma:
        """
        Load an existing vector store.

        Returns:
            Chroma vector store
        """
        vectorstore_path = VECTOR_DB_DIR / VECTOR_DB_NAME

        if not vectorstore_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {vectorstore_path}. "
                "Please run setup_vectorstore.py first."
            )

        print(f"Loading vector store from {vectorstore_path}...")

        vectorstore = Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=self.embeddings,
            collection_name=VECTOR_DB_NAME,
        )

        print(f"Vector store loaded successfully")
        return vectorstore

    def initialize(self, force_recreate: bool = False):
        """
        Initialize the vector store (load or create).

        Args:
            force_recreate: If True, recreate the vector store even if it exists
        """
        vectorstore_path = VECTOR_DB_DIR / VECTOR_DB_NAME

        if force_recreate or not vectorstore_path.exists():
            # Create new vector store
            documents = self.load_documents()
            chunks = self.split_documents(documents)
            self.vectorstore = self.create_vectorstore(chunks)
        else:
            # Load existing vector store
            self.vectorstore = self.load_vectorstore()

    def similarity_search(
        self, query: str, k: int = TOP_K_RESULTS, filter_dict: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")

        results = self.vectorstore.similarity_search(query=query, k=k, filter=filter_dict)

        return results

    def similarity_search_with_score(
        self, query: str, k: int = TOP_K_RESULTS
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of tuples (document, score)
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")

        results = self.vectorstore.similarity_search_with_score(query=query, k=k)

        return results

    def get_retriever(self, **kwargs):
        """
        Get a retriever interface for the vector store.

        Args:
            **kwargs: Additional arguments for retriever configuration

        Returns:
            Retriever object
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")

        return self.vectorstore.as_retriever(search_kwargs={"k": kwargs.get("k", TOP_K_RESULTS)})

    def add_documents(self, documents: List[Document]):
        """
        Add new documents to the existing vector store.

        Args:
            documents: List of documents to add
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")

        chunks = self.split_documents(documents)
        self.vectorstore.add_documents(chunks)
        print(f"Added {len(chunks)} new chunks to vector store")

    def delete_collection(self):
        """Delete the vector store collection."""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            print("Vector store collection deleted")


# Convenience function for quick retrieval
def retrieve_context(query: str, k: int = TOP_K_RESULTS) -> str:
    """
    Retrieve relevant context for a query.

    Args:
        query: Search query
        k: Number of documents to retrieve

    Returns:
        Concatenated context from retrieved documents
    """
    manager = VectorStoreManager()
    manager.initialize()

    results = manager.similarity_search(query, k=k)

    # Combine retrieved documents into context
    context = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in results
        ]
    )

    return context

