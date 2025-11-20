"""
Main execution script for TaskFlow Pro support agent system.

This script:
1. Monitors incoming support emails
2. Processes them through the multi-agent workflow
3. Sends automated responses
"""

import sys
import time
import os
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# SUPPRESS VERBOSE OUTPUT
# ============================================================

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress TensorFlow/CUDA warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA messages

# Suppress HuggingFace progress bars and downloads
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Configure logging to suppress warnings
logging.basicConfig(level=logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
logging.getLogger("spacy").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Configure LangSmith tracing if enabled
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if LANGSMITH_ENABLED and os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "product-support-agents")
    if os.getenv("LANGSMITH_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

# # Add project root to path
# sys.path.insert(0, str(Path(__file__).parent))

from utils.config import validate_config, GMAIL_EMAIL, CHECK_INTERVAL_SECONDS
from utils.email_handler import GmailHandler, MockEmailHandler
from workflows.support_workflow import SupportWorkflow
from utils.metrics import get_tracker, EmailMetrics


def print_banner():
    """Print startup banner."""
    print("TaskFlow Pro - AI Customer Support Automation")
    print(f"Monitoring: {GMAIL_EMAIL}")
    print(f"Check interval: {CHECK_INTERVAL_SECONDS} seconds")
    
    # Show LangSmith status
    if LANGSMITH_ENABLED:
        project_name = os.getenv("LANGSMITH_PROJECT", "product-support-agents")
        print(f" LangSmith tracing enabled (project: {project_name})")
        print(f"  View traces at: https://smith.langchain.com")
    else:
        print("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")


def print_email_details(email, result, processing_time):
    """Print detailed email processing information."""
    print("\n" + "=" * 80)
    print("INCOMING EMAIL")
    print("=" * 80)
    print(f"From: {email.sender}")
    print(f"Subject: {email.subject}")
    print(f"Category: {result.get('category', 'unknown')} | Priority: {result.get('priority', 'unknown')}")
    
    if email.body:
        print(f"\nCustomer Message:")
        print("-" * 80)
        # Show full email body (not truncated)
        print(email.body)
        print("-" * 80)
    
    # Response
    response = result.get("final_response") or result.get("draft_response")
    if response:
        print("\n" + "=" * 80)
        print(" GENERATED RESPONSE")
        print("=" * 80)
        print(response)
        print("-" * 80)
        
        # Show if response was redacted
        if result.get("response_redacted"):
            print("  Note: PII was automatically redacted from this response")
    
    # QA Results (compact)
    qa_score = result.get('qa_score', 0)
    if qa_score > 0:
        print(f"\n Quality Score: {qa_score:.1f}/10 | Approved: {'Yes' if result.get('qa_approved') else 'No'}")
        
        if result.get("qa_issues"):
            print("\n  Issues Found:")
            for issue in result["qa_issues"]:
                print(f"   • {issue}")
    
    # Safety Checks
    safety_violations = result.get("safety_violations", [])
    if safety_violations:
        print("\n  Safety Checks:")
        for violation in safety_violations:
            print(f" {violation}")
    
    # Final Status - CLEAR INDICATOR
    print("\n" + "=" * 80)
    status = result.get("status", "unknown")
    
    if status == "completed_approved":
        if result.get("feedback_saved"):
            print(" ACTION: Feedback Logged (not sent, just saved)")
        else:
            print(" ACTION: EMAIL SENT TO CUSTOMER")
    elif status == "requires_manual_review":
        print("  ACTION: FLAGGED FOR MANUAL REVIEW (email NOT sent)")
    elif status == "completed_skipped":
        print("  ACTION: SKIPPED (no automated response)")
    elif status == "error":
        print(" ACTION: ERROR (email NOT sent)")
    else:
        print(f" ACTION: {status}")
    
    print(f"  Processing Time: {processing_time:.2f}s")
    print("=" * 80)


def detect_escalation(response_text: str) -> bool:
    """
    Detect if response contains escalation language.

    Args:
        response_text: The email response text

    Returns:
        True if escalation detected, False otherwise
    """
    if not response_text:
        return False

    response_lower = response_text.lower()

    # Escalation keywords
    escalation_keywords = [
        "escalat",
        "escalating",
        "escalate to",
        "specialist",
        "specialist will",
        "billing team",
        "technical team",
        "technical specialist",
        "product team",
        "support team",
        "review by",
        "forward to",
        "forwarding to",
        "manual review",
        "need to connect you",
        "reach out to you directly",
        "will contact you",
        "will be in touch",
        "within 24 hours",
        "dedicated",
    ]

    # Check if any escalation keyword is present
    return any(keyword in response_lower for keyword in escalation_keywords)


def process_and_respond(workflow: SupportWorkflow, email_handler: GmailHandler, email) -> dict:
    """
    Process an email and send response if approved.
    
    Returns:
        dict: Processing result with status information
    """
    start_time = time.time()
    
    # Add LangSmith metadata if enabled
    if LANGSMITH_ENABLED:
        try:
            from langsmith import traceable
            from langsmith.run_helpers import get_current_run_tree
        except ImportError:
            pass
    
    # Process email
    result = workflow.process_email(email)
    processing_time = time.time() - start_time
    
    # Create metrics
    metrics = EmailMetrics(
        email_id=email.id,
        category=result.get("category", "unknown"),
        priority=result.get("priority", "unknown"),
        total_time=processing_time,
        qa_score=result.get("qa_score", 0),
        qa_approved=result.get("qa_approved", False),
        revision_count=result.get("revision_count", 0),
        docs_retrieved=len(result.get("rag_sources", [])),
        status=result.get("status", "unknown")
    )
    
    # Track metrics
    get_tracker().add_email_metrics(metrics)
    
    # Add metadata to LangSmith trace
    if LANGSMITH_ENABLED:
        try:
            run = get_current_run_tree()
            if run:
                run.metadata.update({
                    "email_id": email.id,
                    "email_sender": email.sender,
                    "email_subject": email.subject,
                    "category": result.get("category"),
                    "priority": result.get("priority"),
                    "qa_score": result.get("qa_score", 0),
                    "qa_approved": result.get("qa_approved", False),
                    "processing_time_seconds": processing_time,
                    "docs_retrieved": len(result.get("rag_sources", [])),
                    "status": result.get("status"),
                    "revision_count": result.get("revision_count", 0),
                    "needs_review": result.get("status") == "requires_manual_review"
                })
        except Exception:
            pass  # Silently fail if LangSmith is not available
    
    # Print details
    print_email_details(email, result, processing_time)
    
    # Handle based on status
    email_sent = False
    
    if result["status"] == "completed_skipped":
        email_handler.mark_as_read(email.id)
        result["email_sent"] = False
        return result

    if result["status"] == "error":
        result["email_sent"] = False
        return result

    if result["status"] == "requires_manual_review":
        if hasattr(email_handler, "add_label"):
            email_handler.add_label(email.id, "NEEDS_REVIEW")
        result["email_sent"] = False
        return result

    if result["status"] == "completed_approved" and result.get("final_response"):
        final_response = result["final_response"]
        is_escalation = detect_escalation(final_response)

        if not result.get("feedback_saved"):
            email_handler.send_reply(email=email, reply_body=final_response)
            email_sent = True

        if is_escalation:
            if hasattr(email_handler, "add_label"):
                email_handler.add_label(email.id, "NEEDS_REVIEW")
            else:
                email_handler.mark_as_read(email.id)
        else:
            email_handler.mark_as_read(email.id)
    
    result["email_sent"] = email_sent
    return result


def run_continuous_monitoring(workflow: SupportWorkflow, email_handler: GmailHandler):
    """Run continuous email monitoring."""
    print("Starting continuous email monitoring...")
    print("Press Ctrl+C to stop\n")

    processed_count = 0

    try:
        while True:
            new_emails = email_handler.check_new_emails()

            if new_emails:
                print(f"\nFound {len(new_emails)} new email(s)\n")

                for email in new_emails:
                    processed_count += 1
                    print(f"\nProcessing Email #{processed_count}")

                    process_and_respond(workflow, email_handler, email)

                    time.sleep(2)
            else:
                print(f"Checking for emails... (checked at {time.strftime('%H:%M:%S')})", end="\r")

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\nStopping email monitoring...")
        print(f"Total emails processed: {processed_count}")
        
        # Show metrics summary
        get_tracker().print_summary()
        
        print("Goodbye\n")


def run_batch_processing(workflow: SupportWorkflow, email_handler: GmailHandler):
    """Process all current emails once and exit."""
    print("Running batch processing mode...\n")

    emails = email_handler.check_new_emails()

    if not emails:
        print("No new emails to process.\n")
        return

    print(f"Found {len(emails)} email(s) to process\n")

    # Track results
    results_summary = {
        "total": len(emails),
        "sent": 0,
        "manual_review": 0,
        "skipped": 0,
        "errors": 0,
        "feedback_saved": 0
    }

    for i, email in enumerate(emails, 1):
        print(f"\n[{i}/{len(emails)}]")
        result = process_and_respond(workflow, email_handler, email)
        
        # Track status
        if result.get("email_sent"):
            results_summary["sent"] += 1
        elif result.get("feedback_saved"):
            results_summary["feedback_saved"] += 1
        elif result.get("status") == "requires_manual_review":
            results_summary["manual_review"] += 1
        elif result.get("status") == "completed_skipped":
            results_summary["skipped"] += 1
        elif result.get("status") == "error":
            results_summary["errors"] += 1

        if i < len(emails):
            time.sleep(1)

    # Show summary
    print("\n" + "=" * 80)
    print(" BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total Emails Processed: {results_summary['total']}")
    print(f"Emails Sent: {results_summary['sent']}")
    print(f"Feedback Saved: {results_summary['feedback_saved']}")
    print(f"Manual Review: {results_summary['manual_review']}")
    print(f"Skipped: {results_summary['skipped']}")
    print(f"Errors: {results_summary['errors']}")
    print("=" * 80)
    
    # Show metrics summary
    get_tracker().print_summary()


def main():
    """Main entry point."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TaskFlow Pro AI Support Agent System")
    parser.add_argument(
        "--mode",
        choices=["continuous", "batch"],
        default="batch",
        help="Run mode: continuous monitoring or batch processing (default: batch)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Maximum number of response revisions (default: 2)",
    )
    parser.add_argument(
        "--mock-emails",
        action="store_true",
        help="Use mock emails instead of real Gmail (useful for testing/Colab)",
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    try:
        print("Validating configuration...")
        validate_config()
        print("Configuration valid\n")
    except ValueError as e:
        print(f"Configuration error:\n{e}\n")
        sys.exit(1)

    try:
        print("Initializing support system...")

        # Choose email handler based on mode
        if args.mock_emails:
            email_handler = MockEmailHandler(GMAIL_EMAIL)
        else:
            email_handler = GmailHandler(GMAIL_EMAIL)

        workflow = SupportWorkflow(max_revisions=args.max_revisions)
        print("System initialized\n")
    except Exception as e:
        print(f"Initialization error: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    try:
        if args.mode == "continuous":
            run_continuous_monitoring(workflow, email_handler)
        else:
            run_batch_processing(workflow, email_handler)
    except Exception as e:
        print(f"\nError: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
