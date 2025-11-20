"""LangGraph workflow for customer support email automation."""

from typing import TypedDict, Annotated, Sequence, final
from typing_extensions import TypedDict
import operator
import concurrent.futures
from langgraph.graph import StateGraph, END

from agents.classifier import EmailClassifierAgent
from agents.rag_agent import RAGAgent
from agents.response_generator import ResponseGeneratorAgent
from agents.qa_agent import QAAgent
from utils.email_handler import Email
from utils.parallel_utils import ParallelExecutor
from utils.response_cache import ResponseCache
from utils.unified_llm_loader import load_llm
from utils.guardrails_manager import get_guardrails_manager


# Define the state schema
class SupportState(TypedDict):
    """State object for the support workflow."""

    # Email data
    email: Email

    # Classification results
    category: str
    priority: str
    confidence: float
    should_process: bool

    # RAG results
    rag_context: str
    rag_sources: list
    enhanced_queries: dict

    # Response generation
    draft_response: str
    final_response: str

    # QA results
    qa_approved: bool
    qa_score: float
    qa_issues: list
    qa_suggestions: list

    # Workflow control
    revision_count: int
    max_revisions: int
    status: str
    error_message: str
    
    # Guardrails results
    guardrail_violations: list
    guardrail_warnings: list
    response_redacted: bool


class SupportWorkflow:
    """Multi-agent workflow for automated customer support."""

    def __init__(self, max_revisions: int = 0, use_parallel: bool = True, use_cache: bool = True):
        """
        Initialize the support workflow.

        Args:
            max_revisions: Maximum number of times to revise a response (default: 0 = no retries)
            use_parallel: Enable parallel processing for faster execution (default: True)
            use_cache: Enable response caching for FAQ-style emails (default: True)
        """
        self.max_revisions = max_revisions
        self.use_parallel = use_parallel
        self.use_cache = use_cache

        print("Initializing system...")
        self.classifier = EmailClassifierAgent()
        self.rag_agent = RAGAgent(
            use_reranking=True,
            use_query_enhancement=True,
            use_hybrid_search=True,
        )
        self.response_generator = ResponseGeneratorAgent()
        self.qa_agent = QAAgent()

        if use_parallel:
            self.parallel_executor = ParallelExecutor(max_workers=2)

        if use_cache:
            self.response_cache = ResponseCache()
            self.response_cache.clear_expired()
        
        print("System ready\n")

        # Build the workflow graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create graph
        workflow = StateGraph(SupportState)

        # Add nodes
        workflow.add_node("classify", self.classify_email)
        workflow.add_node("check_process", self.check_should_process)
        workflow.add_node("save_feedback", self.save_feedback)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("validate_safety", self.validate_response_safety)
        workflow.add_node("quality_check", self.quality_check)
        workflow.add_node("finalize", self.finalize_response)

        # Add edges
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "check_process")

        # Conditional edge: process, save feedback, or skip
        workflow.add_conditional_edges(
            "check_process",
            self.route_after_classification,
            {"process": "retrieve_context", "feedback": "save_feedback", "skip": "finalize"},
        )

        workflow.add_edge("save_feedback", "finalize")

        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "validate_safety")
        
        # Conditional edge: if safety check passes, go to QA, otherwise finalize
        workflow.add_conditional_edges(
            "validate_safety",
            self.route_after_safety_check,
            {"safe": "quality_check", "blocked": "finalize"},
        )

        # Conditional edge: approve or revise
        workflow.add_conditional_edges(
            "quality_check",
            self.route_after_qa,
            {"approved": "finalize", "revise": "generate_response", "reject": "finalize"},
        )

        workflow.add_edge("finalize", END)

        return workflow.compile()

    # Node functions

    def classify_email(self, state: SupportState) -> SupportState:
        """Classify the incoming email."""
        email = state["email"]
        email_data = {"sender": email.sender, "subject": email.subject, "body": email.body}

        classification = self.classifier.classify(email_data)

        state["category"] = classification.category
        state["priority"] = classification.priority
        state["confidence"] = classification.confidence
        state["should_process"] = self.classifier.should_process(classification)
        state["revision_count"] = 0
        state["max_revisions"] = self.max_revisions
        
        # Initialize guardrails fields
        state["guardrail_violations"] = []
        state["guardrail_warnings"] = []
        state["response_redacted"] = False

        return state

    def check_should_process(self, state: SupportState) -> SupportState:
        """Check if email should be processed."""
        if not state["should_process"]:
            state["status"] = "skipped"
        else:
            state["status"] = "processing"

        return state

    def retrieve_context(self, state: SupportState) -> SupportState:
        """Retrieve relevant context using RAG with metadata filtering."""
        email = state["email"]
        category = state["category"]
        
        # Track enhanced queries
        enhanced_queries = {}

        # For product inquiries, use RAG heavily
        if category == "product_inquiry":
            result = self.rag_agent.answer_question(
                email.body,
                email_subject=email.subject,
                return_sources=True
            )
            state["rag_context"] = result["answer"]
            state["rag_sources"] = result.get("sources", [])
            
            # Extract query info from sources
            if result.get("sources"):
                enhanced_queries["body"] = result["sources"][0].get("query_info", {})
        else:
            # Use parallel retrieval for subject and body separately (faster)
            if self.use_parallel:
                tasks = [
                    {
                        "func": self.rag_agent.retrieve_context,
                        "args": (email.subject,),
                        "kwargs": {"k": 1, "category": category, "email_subject": email.subject},
                    },
                    {
                        "func": self.rag_agent.retrieve_context,
                        "args": (email.body,),
                        "kwargs": {"k": 1, "category": category, "email_subject": email.subject},
                    },
                ]
                results = self.parallel_executor.run_parallel(tasks)
                
                # Extract query info
                if results[0]:
                    enhanced_queries["subject"] = results[0][0].get("query_info", {})
                if results[1]:
                    enhanced_queries["body"] = results[1][0].get("query_info", {})
                
                # Combine results
                docs = results[0] + results[1]
                # Remove duplicates by content
                seen = set()
                unique_docs = []
                for doc in docs:
                    if doc["content"] not in seen:
                        seen.add(doc["content"])
                        unique_docs.append(doc)
                docs = unique_docs[:2]  # Keep top 2
            else:
                docs = self.rag_agent.retrieve_context(
                    email.subject + " " + email.body,
                    k=2,
                    category=category,
                    email_subject=email.subject
                )
                
                # Extract query info
                if docs:
                    enhanced_queries["combined"] = docs[0].get("query_info", {})

            if docs:
                context = "\n\n".join([doc["content"] for doc in docs])
                state["rag_context"] = context
                state["rag_sources"] = docs
            else:
                state["rag_context"] = ""
                state["rag_sources"] = []
        
        # Store enhanced queries in state
        state["enhanced_queries"] = enhanced_queries

        return state

    def _extract_feedback(self, email_body: str, category: str) -> str:
        """Extract the core feedback or feature request from email body using LLM."""
        llm = load_llm(temperature=0.3, max_tokens=200)

        prompt = f"""Extract ONLY the feedback or feature request from this email. Remove greetings, salutations, personal anecdotes, and closing statements. Return just the core feedback or feature suggestion in 1-3 concise sentences.

Category: {category}
Email Body:
{email_body}

Extracted feedback/feature:"""

        try:
            raw_response = llm.invoke(prompt)

            if hasattr(raw_response, "content"):
                extracted = raw_response.content.strip()
            else:
                extracted = str(raw_response).strip()

            # Clean up any artifacts
            extracted = extracted.replace("Extracted feedback/feature:", "").strip()
            extracted = extracted.replace("Feedback:", "").strip()
            extracted = extracted.replace("Feature:", "").strip()

            # Remove quotes if wrapped
            if extracted.startswith('"') and extracted.endswith('"'):
                extracted = extracted[1:-1]

            return extracted
        except Exception as e:
            print(f"Error extracting feedback: {e}")
            # Fallback to first few sentences
            sentences = email_body.split(".")[:3]
            return ". ".join(s.strip() for s in sentences if s.strip()) + "."

    def _extract_name(self, sender_email: str, email_body: str) -> str:
        """Extract name from email sender address or email body."""
        # Try to extract from email body signature first
        lines = email_body.split("\n")
        for line in reversed(lines[-5:]):  # Check last 5 lines
            line = line.strip()
            if line and not line.startswith(("Best", "Thanks", "Regards", "Sincerely", "-")):
                # Simple check for name-like pattern
                if len(line.split()) <= 3 and not "@" in line:
                    return line

        # Fallback to extracting from email address
        name_part = sender_email.split("@")[0]
        # Convert snake_case or camelCase to readable
        name_part = name_part.replace(".", " ").replace("_", " ").replace("-", " ")
        name_part = " ".join(word.capitalize() for word in name_part.split())

        return name_part if name_part else "Customer"

    def save_feedback(self, state: SupportState) -> SupportState:
        """Save feedback or feature requests to log file with extracted content."""
        from pathlib import Path
        from datetime import datetime

        email = state["email"]
        category = state["category"]

        feedback_dir = Path("feedback_logs")
        feedback_dir.mkdir(exist_ok=True)

        # Use single log file that gets appended to
        log_file = feedback_dir / f"{category}_log.txt"

        # Extract name and feedback
        name = self._extract_name(email.sender, email.body)
        feedback = self._extract_feedback(email.body, category)
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format log entry: Name, Date, Feedback
        log_entry = f"{name} | {date} | {feedback}\n"

        # Append to log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

        email_data = {"sender": email.sender, "subject": email.subject, "body": email.body}

        response = self.response_generator.generate_response(
            email_data=email_data, category=category, context=None
        )

        final_response = self.response_generator.add_signature(response)
        state["draft_response"] = final_response
        state["final_response"] = final_response
        state["qa_approved"] = True
        state["qa_score"] = 10.0
        state["status"] = "completed_approved"
        state["feedback_saved"] = True
        state["feedback_file"] = str(log_file)
        state["feedback_entry"] = f"{name} | {feedback[:100]}..."

        return state

    def generate_response(self, state: SupportState) -> SupportState:
        """Generate email response (with caching support)."""
        revision_count = state.get("revision_count", 0)
        email = state["email"]
        category = state["category"]

        # Check cache first (only for first attempt, not revisions)
        if revision_count == 0 and self.use_cache:
            cached = self.response_cache.get(
                email_subject=email.subject,
                email_body=email.body,
                category=category,
                use_fuzzy=True,
            )

            if cached and cached.get("qa_score", 0) >= 7.0:
                # Use cached response
                final_response = cached["response"]
                state["draft_response"] = final_response
                state["qa_score"] = cached["qa_score"]
                state["qa_approved"] = True
                return state

        email_data = {"sender": email.sender, "subject": email.subject, "body": email.body}

        # Use RAG context if available
        context = state.get("rag_context", None)

        # Generate response
        response = self.response_generator.generate_response(
            email_data=email_data, category=category, context=context
        )

        final_response = self.response_generator.add_signature(response)
        state["draft_response"] = final_response

        return state
    
    def validate_response_safety(self, state: SupportState) -> SupportState:
        """Validate response for safety issues using guardrails."""
        guardrails = get_guardrails_manager()
        
        response = state.get("draft_response", "")
        email = state["email"]
        category = state.get("category", "")
        
        # Validate response
        validation_result = guardrails.validate_response(
            response_text=response,
            email_context=email.body,
            category=category
        )
        
        # Store results in state
        state["guardrail_violations"] = validation_result["violations"]
        state["response_redacted"] = validation_result.get("redacted_response") is not None
        
        # If response needs to be redacted, use the redacted version
        if validation_result.get("redacted_response"):
            state["draft_response"] = validation_result["redacted_response"]
        
        # If response should be blocked, flag for manual review
        if validation_result["should_block"]:
            state["status"] = "requires_manual_review"
            state["error_message"] = "Response blocked by safety guardrails"
            
            # Add violation details to QA issues
            violation_messages = [v.message for v in validation_result["violations"]]
            state["qa_issues"] = violation_messages
            state["qa_approved"] = False
            state["qa_score"] = 0.0
        
        return state

    def quality_check(self, state: SupportState) -> SupportState:
        """Perform quality assurance check."""
        email = state["email"]
        email_data = {"sender": email.sender, "subject": email.subject, "body": email.body}

        quick_result = self.qa_agent.quick_check(state["draft_response"])

        # Full QA review
        qa_result = self.qa_agent.review(
            original_email=email_data,
            generated_response=state["draft_response"],
            category=state["category"],
            priority=state["priority"],
        )

        state["qa_approved"] = qa_result.approved
        state["qa_score"] = qa_result.quality_score
        state["qa_issues"] = qa_result.issues
        state["qa_suggestions"] = qa_result.suggestions

        return state

    def finalize_response(self, state: SupportState) -> SupportState:
        """Finalize the workflow."""
        if state["status"] == "skipped":
            state["final_response"] = ""
            state["status"] = "completed_skipped"
        elif state.get("qa_approved", False):
            state["final_response"] = state["draft_response"]
            state["status"] = "completed_approved"

            # Cache the approved response for future use
            if self.use_cache and state.get("revision_count", 0) == 0:
                email = state["email"]
                self.response_cache.set(
                    email_subject=email.subject,
                    email_body=email.body,
                    category=state["category"],
                    response=state["final_response"],
                    qa_score=state.get("qa_score", 0),
                    metadata={
                        "priority": state.get("priority", "medium"),
                        "confidence": state.get("confidence", 0),
                    },
                )
        else:
            state["final_response"] = state["draft_response"]
            state["status"] = "requires_manual_review"

        return state

    # Routing functions

    def route_after_classification(self, state: SupportState) -> str:
        """Route after classification based on category."""
        category = state.get("category", "")

        if category in ["feedback", "feature_request"]:
            return "feedback"
        elif state["should_process"]:
            return "process"
        else:
            return "skip"
    
    def route_after_safety_check(self, state: SupportState) -> str:
        """Route after safety validation."""
        # If status was set to requires_manual_review by guardrails, block
        if state.get("status") == "requires_manual_review":
            return "blocked"
        
        # Check if there are critical violations
        violations = state.get("guardrail_violations", [])
        if any(v.severity == "critical" for v in violations):
            return "blocked"
        
        # Otherwise proceed to QA
        return "safe"

    def route_after_qa(self, state: SupportState) -> str:
        """Route after QA based on approval and revision count."""
        if state["qa_approved"]:
            return "approved"

        # Check if we can revise
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 0)

        if revision_count < max_revisions:
            state["revision_count"] = revision_count + 1
            return "revise"
        else:
            return "reject"

    # Main execution

    def process_email(self, email: Email) -> SupportState:
        """
        Process a single email through the workflow.

        Args:
            email: Email object to process

        Returns:
            Final state after processing
        """
        # Initialize state
        initial_state = SupportState(
            email=email,
            category="",
            priority="",
            confidence=0.0,
            should_process=False,
            rag_context="",
            rag_sources=[],
            draft_response="",
            final_response="",
            qa_approved=False,
            qa_score=0.0,
            qa_issues=[],
            qa_suggestions=[],
            revision_count=0,
            max_revisions=self.max_revisions,
            status="",
            error_message="",
        )

        # Run the workflow
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            import traceback

            traceback.print_exc()
            initial_state["status"] = "error"
            initial_state["error_message"] = str(e)
            return initial_state
