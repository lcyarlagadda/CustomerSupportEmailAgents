"""RAG (Retrieval-Augmented Generation) agent for product documentation queries."""
from typing import List, Dict, Any
from langchain.schema import Document

from utils.config import TOP_K_RESULTS
from utils.vector_store import VectorStoreManager
from utils.unified_llm_loader import load_llm


class RAGAgent:
    """Agent responsible for retrieving and synthesizing information from product documentation."""
    
    def __init__(self):
        """Initialize the RAG agent."""
        # RAG synthesis needs moderate length responses - use 256 tokens
        self.llm = load_llm(temperature=0.7, max_tokens=256)
        
        self.vector_store = VectorStoreManager()
        try:
            self.vector_store.initialize()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Please run setup_vectorstore.py first")
        
        self.synthesis_prompt_template = """You are a helpful support agent for TaskFlow Pro, a project management platform.

Your task is to answer customer questions using the provided documentation context.

Guidelines:
- Only use information from the provided context
- Be clear, concise, and helpful
- If the context doesn't contain enough information, acknowledge this
- Format your response in a friendly, professional tone
- Include specific steps or details when relevant
- If referencing features, mention which plan they're available in if specified

Context from documentation:
{context}

Customer question: {question}

Please provide a helpful, accurate answer based on the documentation context."""
    
    def retrieve_context(
        self, 
        query: str, 
        k: int = TOP_K_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documentation for a query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
        
        Returns:
            List of dictionaries containing document content and metadata
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        retrieved_docs = []
        for doc, score in results:
            retrieved_docs.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "score": score
            })
        
        return retrieved_docs
    
    def synthesize_answer(
        self, 
        question: str, 
        context_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize an answer from retrieved documentation.
        
        Args:
            question: The customer's question
            context_docs: Retrieved documentation chunks
        
        Returns:
            Synthesized answer
        """
        context = "\n\n---\n\n".join([
            f"[Source: {doc['source']}]\n{doc['content']}"
            for doc in context_docs
        ])
        
        prompt = self.synthesis_prompt_template.format(
            context=context,
            question=question
        )
        
        response = self.llm.invoke(prompt)
        
        return response
    
    def answer_question(
        self, 
        question: str, 
        k: int = TOP_K_RESULTS,
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: The customer's question
            k: Number of documents to retrieve
            return_sources: If True, include source documents in response
        
        Returns:
            Dictionary containing answer and optionally sources
        """
        # Retrieve relevant documentation
        context_docs = self.retrieve_context(question, k=k)
        
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
    
    def is_question_answerable(
        self, 
        question: str, 
        threshold: float = 1.5
    ) -> bool:
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


if __name__ == "__main__":
    # Test the RAG agent
    print("Testing RAG Agent...")
    print("=" * 60)
    
    agent = RAGAgent()
    
    # Test questions
    test_questions = [
        "How do I create a new project in TaskFlow Pro?",
        "What's the difference between Professional and Enterprise plans?",
        "How does the Slack integration work?",
        "Can I integrate with GitHub?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        
        try:
            result = agent.answer_question(question, return_sources=True)
            print(f"\nAnswer:")
            print(result["answer"])
            print(f"\n(Used {result['sources_used']} source documents)")
            
            if "sources" in result:
                print("\nSources:")
                for source in result["sources"]:
                    print(f"  - {source['source']} (score: {source['relevance_score']:.4f})")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 60)

