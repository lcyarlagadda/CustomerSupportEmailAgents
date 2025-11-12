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
from pathlib import Path

# Suppress ChromaDB telemetry warnings (must be set before importing chromadb)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Suppress ChromaDB telemetry errors in logs
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

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


def print_email_details(email, result, processing_time):
    """Print detailed email processing information."""
    print("\n" + "=" * 70)
    print("EMAIL")
    print("=" * 70)
    print(f"From: {email.sender}")
    print(f"Subject: {email.subject}")
    if email.body:
        print(f"\nBody:\n{email.body[:500]}{'...' if len(email.body) > 500 else ''}")
    print(f"\nCategory: {result.get('category', 'unknown')} | Priority: {result.get('priority', 'unknown')}")
    
    # Enhanced queries
    if result.get('enhanced_queries'):
        print("\n" + "-" * 70)
        print("ENHANCED QUERIES")
        print("-" * 70)
        for query_type, queries in result['enhanced_queries'].items():
            if queries.get('enhanced'):
                print(f"\n{query_type.upper()}:")
                print(f"  Original: {queries.get('original', 'N/A')}")
                print(f"  Enhanced: {queries.get('enhanced', 'N/A')}")
    
    # Response
    response = result.get("final_response") or result.get("draft_response")
    if response:
        print("\n" + "-" * 70)
        print("RESPONSE")
        print("-" * 70)
        print(response)
    
    # QA Results
    qa_score = result.get('qa_score', 0)
    if qa_score > 0:
        print("\n" + "-" * 70)
        print("QA ASSESSMENT")
        print("-" * 70)
        print(f"Score: {qa_score:.1f}/10")
        print(f"Approved: {'Yes' if result.get('qa_approved') else 'No'}")
        
        if result.get("qa_issues"):
            print("\nIssues Found:")
            for issue in result["qa_issues"]:
                print(f"  - {issue}")
        
        if result.get("qa_suggestions"):
            print("\nSuggestions:")
            for suggestion in result["qa_suggestions"]:
                print(f"  - {suggestion}")
    
    # Final Status
    print("\n" + "-" * 70)
    print("STATUS")
    print("-" * 70)
    status = result.get("status", "unknown")
    if status == "completed_approved":
        if result.get("feedback_saved"):
            print("Feedback Logged")
        else:
            print("Email Sent")
    elif status == "requires_manual_review":
        print("Needs Review")
    elif status == "completed_skipped":
        print("Skipped")
    else:
        print(status)
    
    print(f"Processing Time: {processing_time:.2f}s")
    print("=" * 70)


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


def process_and_respond(workflow: SupportWorkflow, email_handler: GmailHandler, email):
    """Process an email and send response if approved."""
    start_time = time.time()
    
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
    
    # Print details
    print_email_details(email, result, processing_time)
    
    # Handle based on status
    if result["status"] == "completed_skipped":
        email_handler.mark_as_read(email.id)
        return

    if result["status"] == "error":
        return

    if result["status"] == "requires_manual_review":
        if hasattr(email_handler, "add_label"):
            email_handler.add_label(email.id, "NEEDS_REVIEW")
        return

    if result["status"] == "completed_approved" and result.get("final_response"):
        final_response = result["final_response"]
        is_escalation = detect_escalation(final_response)

        if not result.get("feedback_saved"):
            email_handler.send_reply(email=email, reply_body=final_response)

        if is_escalation:
            if hasattr(email_handler, "add_label"):
                email_handler.add_label(email.id, "NEEDS_REVIEW")
            else:
                email_handler.mark_as_read(email.id)
        else:
            email_handler.mark_as_read(email.id)


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

    for i, email in enumerate(emails, 1):
        print(f"\n[{i}/{len(emails)}]")
        process_and_respond(workflow, email_handler, email)

        if i < len(emails):
            time.sleep(1)

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
