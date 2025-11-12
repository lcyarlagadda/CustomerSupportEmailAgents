"""Vector store management for RAG system with hybrid search and smart chunking."""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
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
        
        # BM25 index for hybrid search
        self.bm25_index = None
        self.bm25_docs = []
        self.bm25_doc_ids = []

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
        Split documents into chunks using smart chunking (markdown-aware).
        Then enrich each chunk with metadata based on its actual content.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks with enriched metadata
        """
        print(f"Smart chunking documents...")
        
        # Markdown header splitter for semantic boundaries
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        all_chunks = []
        for doc in documents:
            try:
                # First split by markdown headers
                header_splits = markdown_splitter.split_text(doc.page_content)
                
                # Then apply size-based splitting on large chunks
                for chunk in header_splits:
                    # Preserve original metadata (source file)
                    chunk.metadata.update(doc.metadata)
                    
                    if len(chunk.page_content) > CHUNK_SIZE:
                        # Large chunk - split further
                        sub_chunks = self.text_splitter.split_documents([chunk])
                        for sub_chunk in sub_chunks:
                            sub_chunk.metadata.update(chunk.metadata)
                        all_chunks.extend(sub_chunks)
                    else:
                        # Small chunk - keep as is
                        all_chunks.append(chunk)
            except Exception as e:
                # Fallback to regular splitting if markdown splitting fails
                print(f"Warning: Markdown splitting failed for {doc.metadata.get('source', 'unknown')}, using regular splitting")
                regular_chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(regular_chunks)
        
        print(f"Created {len(all_chunks)} semantically-aware chunks")
        
        # Now enrich each chunk with metadata based on its content
        all_chunks = self._enrich_chunk_metadata(all_chunks)
        print(f"Enriched {len(all_chunks)} chunks with content-based metadata")
        
        return all_chunks
    
    def _enrich_chunk_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Enrich chunks with metadata based on their actual content.
        More accurate than tagging entire documents.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Chunks with enhanced metadata
        """
        for chunk in chunks:
            content_lower = chunk.page_content.lower()
            source = chunk.metadata.get("source", "")
            filename = Path(source).stem.lower() if source else ""
            
            # Determine category based on chunk content (not just filename)
            # Check content keywords first, then fall back to filename
            
            # Check for billing/pricing content
            billing_keywords = ["billing", "payment", "invoice", "subscription", "price", "cost", "refund", "charge", "plan"]
            if any(keyword in content_lower for keyword in billing_keywords):
                chunk.metadata["category"] = "billing"
                chunk.metadata["priority"] = "high"
            # Check for technical/troubleshooting content
            elif any(keyword in content_lower for keyword in ["error", "problem", "issue", "troubleshoot", "fix", "resolve", "not working", "failed"]):
                chunk.metadata["category"] = "technical"
                chunk.metadata["priority"] = "high"
            # Check for integration/API content
            elif any(keyword in content_lower for keyword in ["api", "integration", "webhook", "endpoint", "authentication", "oauth"]):
                chunk.metadata["category"] = "integration"
                chunk.metadata["priority"] = "medium"
            # Check for onboarding/getting started content
            elif any(keyword in content_lower for keyword in ["getting started", "quick start", "first", "create", "setup", "installation"]):
                chunk.metadata["category"] = "onboarding"
                chunk.metadata["priority"] = "medium"
            # Fall back to filename-based categorization
            elif "billing" in filename or "pricing" in filename:
                chunk.metadata["category"] = "billing"
                chunk.metadata["priority"] = "high"
            elif "troubleshooting" in filename or "error" in filename:
                chunk.metadata["category"] = "technical"
                chunk.metadata["priority"] = "high"
            elif "integration" in filename or "api" in filename:
                chunk.metadata["category"] = "integration"
                chunk.metadata["priority"] = "medium"
            elif "getting" in filename or "start" in filename:
                chunk.metadata["category"] = "onboarding"
                chunk.metadata["priority"] = "medium"
            else:
                chunk.metadata["category"] = "general"
                chunk.metadata["priority"] = "low"
            
            # Determine document type based on chunk structure and content
            if any(indicator in content_lower for indicator in ["## problem:", "## solution:", "### problem:", "troubleshooting", "error:"]):
                chunk.metadata["doc_type"] = "troubleshooting"
            elif any(indicator in content_lower for indicator in ["## quick start", "### step", "how to", "guide:"]):
                chunk.metadata["doc_type"] = "guide"
            elif any(indicator in content_lower for indicator in ["endpoint", "request", "response", "parameters", "api reference"]):
                chunk.metadata["doc_type"] = "api_reference"
            elif any(indicator in content_lower for indicator in ["faq", "question:", "frequently asked"]):
                chunk.metadata["doc_type"] = "faq"
            else:
                chunk.metadata["doc_type"] = "documentation"
        
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
            # Build BM25 index for hybrid search
            self._build_bm25_index()
        else:
            # Load existing vector store
            self.vectorstore = self.load_vectorstore()
            # Try to load BM25 index
            self._load_bm25_index()

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

    def _build_bm25_index(self):
        """Build BM25 index for keyword-based retrieval."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("Warning: rank-bm25 not installed. Hybrid search disabled.")
            print("Install with: pip install rank-bm25")
            return
        
        if not self.vectorstore:
            return
        
        print("Building BM25 index for hybrid search...")
        
        # Get all documents from vectorstore
        all_docs = self.vectorstore.get()
        
        if not all_docs or 'documents' not in all_docs:
            print("Warning: No documents found in vectorstore")
            return
        
        self.bm25_docs = all_docs['documents']
        self.bm25_doc_ids = all_docs['ids']
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in self.bm25_docs]
        self.bm25_index = BM25Okapi(tokenized_docs)
        
        # Save BM25 index
        bm25_path = VECTOR_DB_DIR / f"{VECTOR_DB_NAME}_bm25.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'docs': self.bm25_docs,
                'ids': self.bm25_doc_ids,
                'index': self.bm25_index
            }, f)
        
        print(f"BM25 index built with {len(self.bm25_docs)} documents")
    
    def _load_bm25_index(self):
        """Load BM25 index from disk."""
        bm25_path = VECTOR_DB_DIR / f"{VECTOR_DB_NAME}_bm25.pkl"
        
        if not bm25_path.exists():
            print("BM25 index not found, building new index...")
            self._build_bm25_index()
            return
        
        try:
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25_docs = data['docs']
                self.bm25_doc_ids = data['ids']
                self.bm25_index = data['index']
            print("BM25 index loaded")
        except Exception as e:
            print(f"Warning: Could not load BM25 index: {e}")
            print("Building new index...")
            self._build_bm25_index()
    
    def hybrid_search(
        self,
        query: str,
        k: int = TOP_K_RESULTS,
        alpha: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Hybrid search combining semantic (vector) and keyword (BM25) search.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for semantic search (0-1)
                   0.5 = equal weight, 0.7 = favor semantic, 0.3 = favor keyword
            filter_dict: Optional metadata filters
        
        Returns:
            List of (document, combined_score) tuples
        """
        if not self.bm25_index:
            print("BM25 index not available, using semantic search only")
            return self.similarity_search_with_score(query, k)
        
        # Semantic search (get more results for better coverage)
        semantic_results = self.similarity_search_with_score(query, k=k*2)
        
        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize scores
        semantic_scores = {}
        for doc, score in semantic_results:
            # Convert distance to similarity (lower distance = higher similarity)
            semantic_scores[doc.page_content] = 1 / (1 + score)
        
        bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_normalized = {}
        for i, score in enumerate(bm25_scores):
            bm25_normalized[self.bm25_docs[i]] = score / bm25_max
        
        # Combine scores
        combined_scores = {}
        all_content = set(semantic_scores.keys()) | set(bm25_normalized.keys())
        
        for content in all_content:
            semantic_score = semantic_scores.get(content, 0)
            bm25_score = bm25_normalized.get(content, 0)
            combined_scores[content] = alpha * semantic_score + (1 - alpha) * bm25_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Convert back to Document objects
        results = []
        for content, score in sorted_results:
            # Find matching document from semantic results
            for doc, _ in semantic_results:
                if doc.page_content == content:
                    results.append((doc, 1 - score))  # Convert back to distance format
                    break
        
        return results

    def delete_collection(self):
        """Delete the vector store collection."""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            print("Vector store collection deleted")
        
        # Also delete BM25 index
        bm25_path = VECTOR_DB_DIR / f"{VECTOR_DB_NAME}_bm25.pkl"
        if bm25_path.exists():
            bm25_path.unlink()
            print("BM25 index deleted")


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

