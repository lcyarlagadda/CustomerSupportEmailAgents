"""
RAG (Retrieval-Augmented Generation) agent for product documentation queries.

Features:
- Reranking using Cross-Encoder for better accuracy
- Query Enhancement with LLM for better semantic matching
- Better result filtering and scoring
"""

from typing import List, Dict, Any
from langchain.schema import Document

from utils.config import TOP_K_RESULTS
from utils.vector_store import VectorStoreManager
from utils.unified_llm_loader import load_llm


class RAGAgent:
    """RAG agent with reranking and query enhancement."""
    
    def __init__(self, use_reranking: bool = True, use_query_enhancement: bool = True):
        """
        Initialize RAG agent.
        
        Args:
            use_reranking: Enable cross-encoder reranking (recommended)
            use_query_enhancement: Enable LLM query enhancement (recommended)
        """
        self.use_reranking = use_reranking
        self.use_query_enhancement = use_query_enhancement
        
        print("Initializing RAG Agent...")
        
        # LLM for answer synthesis
        self.llm = load_llm(temperature=0.7, max_tokens=256)
        
        # Vector store
        self.vector_store = VectorStoreManager()
        try:
            self.vector_store.initialize()
            print("Vector store loaded")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Please run setup_vectorstore.py first")
        
        # Load reranker if enabled
        self.reranker = None
        if use_reranking:
            try:
                from sentence_transformers import CrossEncoder
                print("Loading reranker model (one-time download ~80MB)...")
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("Reranker loaded")
            except ImportError:
                print("Warning: sentence-transformers not found. Reranking disabled.")
                print("Install with: pip install sentence-transformers")
                self.use_reranking = False
            except Exception as e:
                print(f"Warning: Could not load reranker: {e}")
                self.use_reranking = False
        
        self.synthesis_prompt_template = """You are a helpful support agent for TaskFlow Pro, a project management platform.

Your task is to answer customer questions using the provided documentation context.

Guidelines:
- Only use information from the provided context
- Be clear, concise, and helpful
- If the context doesn't contain enough information, acknowledge this
- Format your response in a friendly, professional tone
- Include specific steps or details when relevant

Context from documentation:
{context}

Customer question: {question}

Please provide a helpful, accurate answer based on the documentation context."""
    
    def enhance_query(self, query: str, email_subject: str = "") -> str:
        """
        Enhance query for better retrieval using LLM.
        
        Rewrites vague queries to be more specific and searchable.
        Example: "it's broken" → "application not loading troubleshooting"
        
        Args:
            query: Original user query
            email_subject: Email subject for additional context
            
        Returns:
            Enhanced query
        """
        if not self.use_query_enhancement:
            return query
        
        # Skip enhancement for already clear queries
        if len(query.split()) > 8 and '?' in query:
            return query
        
        context_line = f"Email subject: {email_subject}\n" if email_subject else ""
        
        prompt = f"""{context_line}Rewrite this customer question to be more specific and searchable for documentation lookup.

Original question: {query}

Requirements:
1. Be more specific and detailed
2. Include relevant technical terms
3. Make it self-contained
4. Focus on the core issue

Return ONLY the rewritten question (one sentence):"""
        
        try:
            llm = load_llm(temperature=0.3, max_tokens=100)
            response = llm.invoke(prompt)
            enhanced = response.content if hasattr(response, 'content') else str(response)
            enhanced = enhanced.strip()
            
            # Basic validation
            if len(enhanced) > 10 and len(enhanced) < 200:
                return enhanced
            else:
                return query
        except Exception as e:
            print(f"  Query enhancement failed: {e}")
            return query  # Fallback to original
    
    def retrieve_context(
        self,
        query: str,
        k: int = TOP_K_RESULTS,
        email_subject: str = "",
        use_enhancement: bool = None,
        use_reranking: bool = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documentation with enhancements.
        
        Args:
            query: Search query
            k: Number of documents to return
            email_subject: Email subject for context
            use_enhancement: Override instance setting for query enhancement
            use_reranking: Override instance setting for reranking
            
        Returns:
            List of relevant document dictionaries
        """
        # Allow per-call overrides
        do_enhance = use_enhancement if use_enhancement is not None else self.use_query_enhancement
        do_rerank = use_reranking if use_reranking is not None else self.use_reranking
        
        # Step 1: Enhance query
        enhanced_query = query
        if do_enhance:
            enhanced_query = self.enhance_query(query, email_subject)
            if enhanced_query != query:
                print(f"  Original: {query[:60]}...")
                print(f"  Enhanced: {enhanced_query[:60]}...")
        
        # Step 2: Retrieve documents (get more for reranking)
        initial_k = k * 3 if do_rerank and self.reranker else k
        
        try:
            results = self.vector_store.similarity_search_with_score(enhanced_query, k=initial_k)
        except Exception as e:
            print(f"  Retrieval error: {e}")
            return []
        
        if not results:
            return []
        
        # Step 3: Rerank if enabled
        if do_rerank and self.reranker and len(results) > k:
            print(f"  Reranking {len(results)} → {k} documents...")
            results = self._rerank_results(enhanced_query, results, k)
        
        # Step 4: Format results
        retrieved_docs = []
        for doc, score in results[:k]:
            retrieved_docs.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "score": float(score),
            })
        
        return retrieved_docs
    
    def _rerank_results(
        self,
        query: str,
        results: List[tuple[Document, float]],
        k: int
    ) -> List[tuple[Document, float]]:
        """
        Rerank results using cross-encoder for better accuracy.
        
        Cross-encoders compute query-document interaction scores,
        which are more accurate than cosine similarity.
        
        Args:
            query: Search query
            results: Initial retrieval results
            k: Number of top results to keep
            
        Returns:
            Reranked results (top k)
        """
        # Prepare query-document pairs
        docs = [doc for doc, _ in results]
        pairs = [(query, doc.page_content) for doc in docs]
        
        # Get rerank scores (higher is better)
        try:
            rerank_scores = self.reranker.predict(pairs)
        except Exception as e:
            print(f"  Reranking failed: {e}, using original order")
            return results[:k]
        
        # Combine documents with rerank scores and sort
        doc_score_pairs = list(zip(docs, rerank_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Convert back to (doc, score) format
        # Convert rerank scores to distance-like format for consistency
        reranked = [(doc, 1 - float(score)) for doc, score in doc_score_pairs]
        
        return reranked[:k]
    
    def synthesize_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Synthesize an answer from retrieved documentation.
        
        Args:
            question: The customer's question
            context_docs: Retrieved documentation chunks
            
        Returns:
            Synthesized answer
        """
        if not context_docs:
            return "I don't have sufficient documentation to answer this question accurately."
        
        context = "\n\n---\n\n".join([
            f"[Source: {doc['source']}]\n{doc['content']}" 
            for doc in context_docs
        ])
        
        prompt = self.synthesis_prompt_template.format(
            context=context,
            question=question
        )
        
        raw_response = self.llm.invoke(prompt)
        
        if hasattr(raw_response, "content"):
            response = raw_response.content
        else:
            response = str(raw_response)
        
        return response
    
    def answer_question(
        self,
        question: str,
        k: int = TOP_K_RESULTS,
        email_subject: str = "",
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: The customer's question
            k: Number of documents to retrieve
            email_subject: Email subject for context
            return_sources: If True, include source documents in response
            
        Returns:
            Dictionary containing answer and optionally sources
        """
        # Retrieve with enhancements
        context_docs = self.retrieve_context(question, k, email_subject)
        
        # Synthesize answer
        answer = self.synthesize_answer(question, context_docs)
        
        result = {
            "answer": answer,
            "sources_used": len(context_docs)
        }
        
        if return_sources:
            result["sources"] = [
                {
                    "source": doc["source"],
                    "relevance_score": doc["score"]
                }
                for doc in context_docs
            ]
        
        return result
    
    def is_question_answerable(self, question: str, threshold: float = 1.5) -> bool:
        """
        Determine if a question can be answered from documentation.
        
        Args:
            question: The question to check
            threshold: Maximum score threshold (lower is better)
            
        Returns:
            True if question is answerable, False otherwise
        """
        # Get top result with score
        results = self.vector_store.similarity_search_with_score(question, k=1)
        
        if not results:
            return False
        
        _, score = results[0]
        return score < threshold

