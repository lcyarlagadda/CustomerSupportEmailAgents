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

        print("Initializing agents...")
        self.classifier = EmailClassifierAgent()
        self.rag_agent = RAGAgent()
        self.response_generator = ResponseGeneratorAgent()
        self.qa_agent = QAAgent()
        print("All agents initialized\n")

        if use_parallel:
            self.parallel_executor = ParallelExecutor(max_workers=2)
            print("Parallel processing enabled\n")

        if use_cache:
            self.response_cache = ResponseCache()
            self.response_cache.clear_expired()
            print("Response caching enabled\n")

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
        workflow.add_edge("generate_response", "quality_check")

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
        print("Classifying email...")

        email = state["email"]
        email_data = {"sender": email.sender, "subject": email.subject, "body": email.body}

        classification = self.classifier.classify(email_data)

        print(f"Category: {classification.category}")
        print(f"Priority: {classification.priority}")
        print(f"Confidence: {classification.confidence:.2f}")

        state["category"] = classification.category
        state["priority"] = classification.priority
        state["confidence"] = classification.confidence
        state["should_process"] = self.classifier.should_process(classification)
        state["revision_count"] = 0
        state["max_revisions"] = self.max_revisions

        return state

    def check_should_process(self, state: SupportState) -> SupportState:
        """Check if email should be processed."""
        if not state["should_process"]:
            print(f"Skipping email (category: {state['category']})")
            state["status"] = "skipped"
        else:
            print("Email will be processed")
            state["status"] = "processing"

        return state

    def retrieve_context(self, state: SupportState) -> SupportState:
        """Retrieve relevant context using RAG (optimized with parallel processing)."""
        print("\nRetrieving relevant documentation...")

        email = state["email"]
        category = state["category"]

        # For product inquiries, use RAG heavily
        if category == "product_inquiry":
            result = self.rag_agent.answer_question(email.body, return_sources=True)
            state["rag_context"] = result["answer"]
            state["rag_sources"] = result.get("sources", [])
            print(f"Retrieved context from {result['sources_used']} sources")
        else:
            # Use parallel retrieval for subject and body separately (faster)
            if self.use_parallel:
                tasks = [
                    {
                        "func": self.rag_agent.retrieve_context,
                        "args": (email.subject,),
                        "kwargs": {"k": 1},
                    },
                    {
                        "func": self.rag_agent.retrieve_context,
                        "args": (email.body,),
                        "kwargs": {"k": 1},
                    },
                ]
                results = self.parallel_executor.run_parallel(tasks)
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
                docs = self.rag_agent.retrieve_context(email.subject + " " + email.body, k=2)

            if docs:
                context = "\n\n".join([doc["content"] for doc in docs])
                state["rag_context"] = context
                state["rag_sources"] = docs
                print(f"Retrieved {len(docs)} relevant documentation sections")
            else:
                state["rag_context"] = ""
                state["rag_sources"] = []
                print("No specific documentation found")

        return state

    def save_feedback(self, state: SupportState) -> SupportState:
        """Save feedback or feature requests to file for review."""
        from pathlib import Path
        from datetime import datetime

        email = state["email"]
        category = state["category"]

        feedback_dir = Path("feedback_logs")
        feedback_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = feedback_dir / f"{category}_{timestamp}.txt"

        content = f"""
        Category: {category.upper()}
        Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        From: {email.sender}
        Subject: {email.subject}
        Priority: {state.get('priority', 'N/A')}

        EMAIL CONTENT:
        {email.body}

        ---
        Saved for review by {category.replace('_', ' ').title()} Team
        """

        filename.write_text(content.strip())
        print(f"\n{category.replace('_', ' ').title()} saved to: {filename}")

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

        print("\n" + "─" * 70)
        print("RESPONSE:")
        print("─" * 70)
        print(final_response)
        print("─" * 70)

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

                print("using cached response")
                print(final_response)

                return state

        # Generate new response
        if revision_count > 0:
            print(f"\nGenerating revised response (attempt {revision_count + 1})...")
        else:
            print("\nGenerating response...")

        email_data = {"sender": email.sender, "subject": email.subject, "body": email.body}

        # Use RAG context if available
        context = state.get("rag_context", None)

        # Generate response
        response = self.response_generator.generate_response(
            email_data=email_data, category=category, context=context
        )

        final_response = self.response_generator.add_signature(response)

        state["draft_response"] = final_response
        print("Response generated")

        # Show response preview
        print("Respose")
        print(final_response)

        return state

    def quality_check(self, state: SupportState) -> SupportState:
        """Perform quality assurance check."""
        print("\nPerforming quality check...")

        email = state["email"]
        email_data = {"sender": email.sender, "subject": email.subject, "body": email.body}

        quick_result = self.qa_agent.quick_check(state["draft_response"])
        if not quick_result["passed"]:
            print("Quick check failed:")
            for issue in quick_result["issues"]:
                print(f"  - {issue}")

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

        print(f"Quality Score: {qa_result.quality_score:.1f}/10")
        print(f"Approved: {qa_result.approved}")

        if not qa_result.approved and qa_result.issues:
            print("Issues found:")
            for issue in qa_result.issues[:3]:
                print(f"  - {issue}")

        return state

    def finalize_response(self, state: SupportState) -> SupportState:
        """Finalize the workflow."""
        if state["status"] == "skipped":
            state["final_response"] = ""
            state["status"] = "completed_skipped"
        elif state.get("qa_approved", False):
            state["final_response"] = state["draft_response"]
            state["status"] = "completed_approved"
            print("\nResponse approved and ready to send")

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
            print("\nResponse generated but requires manual review before sending")
            print(f"Issues: {', '.join(state.get('qa_issues', ['Quality threshold not met']))}")

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

    def route_after_qa(self, state: SupportState) -> str:
        """Route after QA based on approval and revision count."""
        if state["qa_approved"]:
            return "approved"

        # Check if we can revise
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 0)

        if revision_count < max_revisions:
            state["revision_count"] = revision_count + 1
            print(
                f"QA failed - Retrying (attempt {state['revision_count'] + 1}/{max_revisions + 1})..."
            )
            return "revise"
        else:
            # Skip retry, mark for manual review
            if max_revisions == 0:
                print("QA check failed - Marking for manual review (no retries enabled)")
            else:
                print(
                    f"QA check failed after {max_revisions} revision(s) - Marking for manual review"
                )
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
        print("=" * 70)
        print(f" Processing Email from {email.sender}")
        print(f"   Subject: {email.subject}")
        print("=" * 70)

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
            print(f"\nError processing email: {e}")
            import traceback

            traceback.print_exc()
            initial_state["status"] = "error"
            initial_state["error_message"] = str(e)
            return initial_state
