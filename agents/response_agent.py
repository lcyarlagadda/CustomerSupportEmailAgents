"""
Unified Response Agent - Combines RAG and Response Generation

This agent handles the complete response pipeline:
1. Query enhancement for better retrieval
2. Document retrieval with reranking
3. Category-specific response generation
4. Email formatting and cleanup

Replaces: RAGAgent + ResponseGeneratorAgent
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document

from utils.config import TOP_K_RESULTS
from utils.vector_store import VectorStoreManager
from utils.unified_llm_loader import load_llm


class ResponseAgent:
    """
    Unified agent for document retrieval and response generation.
    
    Combines RAG capabilities (retrieval, reranking, query enhancement)
    with response generation (category-specific prompts, formatting).
    """
    
    def __init__(
        self,
        use_reranking: bool = True,
        use_query_enhancement: bool = True,
        use_hybrid_search: bool = True
    ):
        """
        Initialize the unified response agent.
        
        Args:
            use_reranking: Enable cross-encoder reranking (recommended)
            use_query_enhancement: Enable LLM query enhancement (recommended)
            use_hybrid_search: Enable hybrid semantic + keyword search (recommended)
        """
        self.use_reranking = use_reranking
        self.use_query_enhancement = use_query_enhancement
        self.use_hybrid_search = use_hybrid_search
        
        # LLM for response generation
        self.llm = load_llm(temperature=0.8, max_tokens=400)
        
        # Vector store for document retrieval
        self.vector_store = VectorStoreManager()
        try:
            self.vector_store.initialize()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Please run setup_vectorstore.py first")
        
        # Load reranker if enabled
        self.reranker = None
        if use_reranking:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✓ Response Agent: Reranking enabled")
            except ImportError:
                print("⚠ Response Agent: sentence-transformers not found, reranking disabled")
                self.use_reranking = False
            except Exception as e:
                print(f"⚠ Response Agent: Could not load reranker: {e}")
                self.use_reranking = False
        
        # Response format template
        self.response_format = """
        REQUIRED FORMAT (follow exactly):
        
        1. Salutation: "Hi [Name]," or "Hello [Name],"
        
        2. Acknowledgment:
        - If issue/problem: Acknowledge and apologize
        - If feedback/suggestion/request: Appreciate their input
        
        3. Main content: [Your specific response]
        
        4. Closing: "Hope you have a great day!"
        
        5. DO NOT include signature - it will be added automatically
        """
        
        # Category-specific prompts (from ResponseGeneratorAgent)
        self.prompts = {
            "technical_support": """You are a professional support agent for TaskFlow Pro. A customer needs technical help.

Customer Email:
Subject: {subject}
Body: {body}

Documentation Available: {context}

{format_instructions}

SPECIFIC INSTRUCTIONS:
- Acknowledge their technical issue and apologize for the inconvenience
- If documentation contains solution: Provide clear step-by-step instructions
- If documentation does NOT contain solution: Apologize and explain you are escalating to a technical specialist who will contact them within 24 hours
- Never make up information or promise things not in the documentation
- Stay calm and professional regardless of customer tone
- Keep response 200-300 words

Write ONLY the email body (no signature):""",
            
            "product_inquiry": """You are a professional support agent for TaskFlow Pro. A customer has a product question.

Customer Email:
Subject: {subject}
Body: {body}

Documentation Available: {context}

{format_instructions}

SPECIFIC INSTRUCTIONS:
- Thank them for their question
- If documentation has the answer: Provide clear, accurate information
- If NOT in documentation: Apologize and explain you are escalating to the product team for accurate information, they will respond within 24 hours
- Never make up features, pricing, or capabilities
- Stay professional and informative
- Keep response 200-300 words

Write ONLY the email body (no signature):""",
            
            "billing": """You are a professional support agent for TaskFlow Pro. A customer has a billing concern.

Customer Email:
Subject: {subject}
Body: {body}

{format_instructions}

SPECIFIC INSTRUCTIONS:
- Acknowledge their billing concern and sincerely apologize for any inconvenience
- Explain that billing matters require direct database access and specialist review
- Assure them you are escalating to the billing team who will contact them within 24 hours with a resolution
- Never attempt to look up account details, process refunds, or make billing promises
- Stay empathetic and reassuring
- Keep response 150-250 words

Write ONLY the email body (no signature):""",
            
            "feature_request": """You are a professional support agent for TaskFlow Pro. A customer has suggested a feature.

Customer Email:
Subject: {subject}
Body: {body}

{format_instructions}

SPECIFIC INSTRUCTIONS:
- Thank them sincerely for their feature suggestion
- Acknowledge the value of their input
- Explain their request has been saved and will be reviewed by the product team
- Do NOT promise implementation or timelines
- Stay professional and appreciative
- Keep response brief: 100-150 words

Write ONLY the email body (no signature):""",
            
            "feedback": """You are a professional support agent for TaskFlow Pro. A customer provided feedback.

Customer Email:
Subject: {subject}
Body: {body}

{format_instructions}

SPECIFIC INSTRUCTIONS:
- If positive feedback: Thank them sincerely for their kind words
- If negative feedback: Acknowledge their concerns and apologize professionally, never defensive
- Explain their feedback has been saved and will be reviewed by leadership
- Stay calm and professional even if customer is frustrated
- Keep response brief: 100-150 words

Write ONLY the email body (no signature):""",
        }
    
    # ============================================================
    # RETRIEVAL METHODS (from RAGAgent)
    # ============================================================
    
    def enhance_query(self, query: str, email_subject: str = "") -> str:
        """
        Enhance query for better retrieval using LLM.
        
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
            
            if len(enhanced) > 10 and len(enhanced) < 200:
                return enhanced
            else:
                return query
        except Exception:
            return query
    
    def retrieve_context(
        self,
        query: str,
        k: int = TOP_K_RESULTS,
        email_subject: str = "",
        category: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documentation with enhancements.
        
        Args:
            query: Search query
            k: Number of documents to return
            email_subject: Email subject for context
            category: Filter by document category
            
        Returns:
            List of relevant document dictionaries
        """
        # Step 1: Enhance query
        enhanced_query = query
        query_info = {"original": query, "enhanced": None}
        if self.use_query_enhancement:
            enhanced_query = self.enhance_query(query, email_subject)
            if enhanced_query != query:
                query_info["enhanced"] = enhanced_query
        
        # Step 2: Apply category filtering
        filter_dict = None
        if category:
            category_map = {
                "technical_support": "technical",
                "product_inquiry": "general",
                "billing": "billing",
                "integration": "integration",
            }
            doc_category = category_map.get(category, category)
            filter_dict = {"category": doc_category}
        
        # Step 3: Retrieve documents
        initial_k = k * 3 if self.use_reranking and self.reranker else k
        
        try:
            if self.use_hybrid_search and hasattr(self.vector_store, 'hybrid_search'):
                results = self.vector_store.hybrid_search(
                    enhanced_query,
                    k=initial_k,
                    alpha=0.7,
                    filter_dict=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    enhanced_query,
                    k=initial_k
                )
        except Exception:
            return []
        
        if not results:
            return []
        
        # Step 4: Rerank if enabled
        if self.use_reranking and self.reranker and len(results) > k:
            results = self._rerank_results(enhanced_query, results, k)
        
        # Step 5: Format results
        retrieved_docs = []
        for doc, score in results[:k]:
            retrieved_docs.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "category": doc.metadata.get("category", "unknown"),
                "score": float(score),
                "query_info": query_info,
            })
        
        return retrieved_docs
    
    def _rerank_results(
        self,
        query: str,
        results: List[tuple[Document, float]],
        k: int
    ) -> List[tuple[Document, float]]:
        """Rerank results using cross-encoder."""
        docs = [doc for doc, _ in results]
        pairs = [(query, doc.page_content) for doc in docs]
        
        try:
            rerank_scores = self.reranker.predict(pairs)
        except Exception:
            return results[:k]
        
        doc_score_pairs = list(zip(docs, rerank_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to distance-like format
        reranked = [(doc, 1 - float(score)) for doc, score in doc_score_pairs]
        
        return reranked[:k]
    
    # ============================================================
    # RESPONSE GENERATION METHODS (from ResponseGeneratorAgent)
    # ============================================================
    
    def generate_response(
        self,
        email_data: Dict[str, Any],
        category: str,
        context: Optional[str] = None,
        context_docs: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a complete email response.
        
        Args:
            email_data: Dictionary with email information
            category: Email category
            context: Optional pre-formatted context string
            context_docs: Optional list of retrieved documents (will be formatted)
            
        Returns:
            Generated email response (without signature)
        """
        # Get appropriate prompt for category
        prompt = self.prompts.get(category)
        
        if not prompt:
            prompt = self._get_general_prompt()
        
        # Prepare context
        if context:
            context_text = context
        elif context_docs:
            # Format documents into context
            context_text = self._format_context_from_docs(context_docs)
        else:
            context_text = "No additional context provided. Use your general knowledge of TaskFlow Pro."
        
        # Format prompt with data
        if "{format_instructions}" in prompt:
            formatted_prompt = prompt.format(
                sender=email_data.get("sender", "Valued Customer"),
                subject=email_data.get("subject", "Your inquiry"),
                body=email_data.get("body", ""),
                context=context_text,
                format_instructions=self.response_format,
            )
        else:
            formatted_prompt = prompt.format(
                sender=email_data.get("sender", "Valued Customer"),
                subject=email_data.get("subject", "Your inquiry"),
                body=email_data.get("body", ""),
                context=context_text,
            )
        
        # Generate response
        raw_response = self.llm.invoke(formatted_prompt)
        
        if hasattr(raw_response, "content"):
            response_text = raw_response.content
        else:
            response_text = str(raw_response)
        
        response = self._clean_response(response_text, email_data)
        
        return response
    
    def _format_context_from_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        if not docs:
            return "No relevant documentation found."
        
        context_parts = []
        for doc in docs:
            context_parts.append(f"[Source: {doc['source']}]\n{doc['content']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _clean_response(self, raw_response: str, email_data: dict) -> str:
        """Clean up LLM response."""
        import re
        
        # Find start of actual response
        start_markers = ["Dear ", "Hi ", "Hello ", "Thank you for", "Thanks for"]
        
        response = raw_response
        
        for marker in start_markers:
            if marker in response:
                idx = response.find(marker)
                if idx > 100:
                    response = response[idx:]
                break
        
        # Remove signatures (we'll add our own)
        signature_markers = [
            "Thanks,", "Best regards,", "Sincerely,", "Best,",
            "Warm regards,", "Kind regards,", "Regards,", "Thank you,"
        ]
        
        first_sig_pos = len(response)
        for marker in signature_markers:
            pos = response.find(marker)
            if pos > 0 and pos < first_sig_pos:
                first_sig_pos = pos
        
        if first_sig_pos < len(response):
            response = response[:first_sig_pos].rstrip()
        
        # Remove placeholder text
        response = re.sub(r"\[Your [^\]]+\]", "", response)
        response = re.sub(r"\(optional\)", "", response)
        
        # Remove meta-commentary
        meta_patterns = [
            r"\n\nNote:.*$", r"\n\nPlease note.*$", r"\n\nPlease let me know.*$",
            r"\n\nThis response.*$", r"\n\nThe response.*$", r"\n\nFeel free to.*$",
            r"\n\nIf you need any modifications.*$", r"\n\n---.*$",
        ]
        
        for pattern in meta_patterns:
            response = re.sub(pattern, "", response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up whitespace
        lines = response.split("\n")
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.lstrip(" \t\u00a0").rstrip(" \t")
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
        
        response = "\n".join(cleaned_lines)
        response = re.sub(r"\n{3,}", "\n\n", response)
        
        return response.strip()
    
    def _get_general_prompt(self) -> str:
        """Get a general-purpose response prompt."""
        return """Customer Email: {subject}
{body}

Available Info: {context}

Write ONLY the email body (200-400 words). Start with greeting, end BEFORE signature.
DO NOT include: [placeholders], notes, explanations, signature, or "please let me know"."""
    
    def add_signature(self, email_body: str) -> str:
        """
        Add professional signature to the email.
        
        Args:
            email_body: The email body content
            
        Returns:
            Email with signature
        """
        signature_check = ["Thanks,", "TaskFlow Pro Team", "support@taskflowpro.com"]
        if any(sig in email_body for sig in signature_check):
            # Already has signature, clean it up
            lines = email_body.split("\n")
            cleaned_lines = []
            for line in lines:
                cleaned_line = line.lstrip(" \t\u00a0").rstrip(" \t")
                cleaned_lines.append(cleaned_line)
            return "\n".join(cleaned_lines)
        
        # Clean up body
        lines = email_body.split("\n")
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.lstrip(" \t\u00a0").rstrip(" \t")
            cleaned_lines.append(cleaned_line)
        email_body = "\n".join(cleaned_lines)
        email_body = email_body.rstrip()
        
        # Ensure proper spacing before signature
        if not email_body.endswith("\n\n"):
            if email_body.endswith("\n"):
                email_body += "\n"
            else:
                email_body += "\n\n"
        
        signature = "Thanks,\nTaskFlow Pro Team\nsupport@taskflowpro.com"
        
        return email_body + signature
    
    # ============================================================
    # UNIFIED HIGH-LEVEL METHODS
    # ============================================================
    
    def generate_with_retrieval(
        self,
        email_data: Dict[str, Any],
        category: str,
        k: int = TOP_K_RESULTS
    ) -> Dict[str, Any]:
        """
        Complete pipeline: retrieve docs → generate response.
        
        Args:
            email_data: Email information
            category: Email category
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve relevant documentation
        query = f"{email_data.get('subject', '')} {email_data.get('body', '')}"
        docs = self.retrieve_context(
            query=query,
            k=k,
            email_subject=email_data.get('subject', ''),
            category=category
        )
        
        # Generate response using retrieved docs
        response = self.generate_response(
            email_data=email_data,
            category=category,
            context_docs=docs
        )
        
        return {
            "response": response,
            "docs_retrieved": len(docs),
            "sources": [doc["source"] for doc in docs],
            "enhanced_queries": docs[0]["query_info"] if docs else {}
        }

