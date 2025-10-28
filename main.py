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


def print_banner():
    """Print startup banner."""
    print("TaskFlow Pro - AI Customer Support Automation")
    print(f"Monitoring: {GMAIL_EMAIL}")
    print(f"Check interval: {CHECK_INTERVAL_SECONDS} seconds")


def print_email_details(email):
    """Print original email details."""
    print("ORIGINAL EMAIL")
    print(f"From:     {email.sender}")
    print(f"Subject:  {email.subject}")
    print(f"Received: {email.received_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Body:")
    print(email.body[:300] + "..." if len(email.body) > 300 else email.body)


def print_classification(state: dict):
    """Print classification results."""
    print("AI CLASSIFICATION")
    print(f"Category:    {state['category'].upper()}")
    print(f"Priority:    {state['priority'].upper()}")
    print(f"Confidence:  {state['confidence']:.1%}")
    print(f"Reasoning:   {state.get('reasoning', 'N/A')[:100]}...")


def print_response(state: dict):
    """Print generated response."""
    response = state.get("final_response") or state.get("draft_response")
    if response:
        print("FINAL GENERATED RESPONSE")
        print(response)


def print_qa_results(state: dict):
    """Print QA results."""
    print("QUALITY ASSURANCE RESULTS")
    print(f"Quality Score: {state.get('qa_score', 0):.1f}/10")
    print(f"Status:        {'APPROVED' if state.get('qa_approved') else 'NEEDS REVISION'}")

    if state.get("qa_issues"):
        print(f"\nIssues Found:")
        for issue in state["qa_issues"]:
            print(f"  - {issue}")

    if state.get("qa_suggestions"):
        print(f"\nSuggestions:")
        for suggestion in state["qa_suggestions"]:
            print(f"  - {suggestion}")

    if state.get("tone_assessment"):
        print(f"\nTone: {state['tone_assessment']}")

    print("=" * 70 + "\n")


def print_final_action(state: dict):
    """Print final action taken."""
    print("FINAL ACTION")

    status = state["status"]
    if status == "completed_skipped":
        print(f"SKIPPED - Category: {state['category']}")
        print("  Reason: Unrelated to product support")
    elif status == "error":
        print(f"ERROR - {state.get('error_message', 'Unknown error')}")
    elif status == "completed_approved":
        print("SENT - Email response sent to customer")
    elif status == "requires_manual_review":
        print("REQUIRES MANUAL REVIEW")
        print("  Reason: QA check did not pass")
        print("  Action: Review and edit response before sending")
    else:
        print(f"PENDING - Status: {status}")

    print("=" * 70 + "\n")


def process_and_respond(workflow: SupportWorkflow, email_handler: GmailHandler, email):
    """Process an email and send response if approved."""

    # 1. Show original email
    print_email_details(email)

    # 2. Process through workflow
    result = workflow.process_email(email)

    # 3. Show classification
    print_classification(result)

    # 4. Show generated response (if any)
    if result.get("final_response"):
        print_response(result)

    # 5. Show QA results
    print_qa_results(result)

    # 6. Show final action
    print_final_action(result)

    # 7. Execute action
    if result["status"] == "completed_skipped":
        email_handler.mark_as_read(email.id)
        return

    if result["status"] == "error":
        return

    if result["status"] == "requires_manual_review":
        # Don't send, mark for manual review
        if hasattr(email_handler, "add_label"):
            email_handler.add_label(email.id, "NEEDS_REVIEW")
        print("Email marked for manual review (not sent automatically)\n")
        return

    if result["status"] == "completed_approved" and result.get("final_response"):
        email_handler.send_reply(email=email, reply_body=result["final_response"])
        email_handler.mark_as_read(email.id)
        print("Email sent successfully\n")


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
                    print(f"\n{'=' * 70}")
                    print(f"Processing Email #{processed_count}")
                    print(f"{'=' * 70}\n")

                    process_and_respond(workflow, email_handler, email)

                    time.sleep(2)
            else:
                print(f"Checking for emails... (checked at {time.strftime('%H:%M:%S')})", end="\r")

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("Stopping email monitoring...")
        print(f"Total emails processed: {processed_count}")
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
        print(f"Processing Email {i}/{len(emails)}")

        process_and_respond(workflow, email_handler, email)

        if i < len(emails):
            time.sleep(2)

    print("Batch processing complete")
    print(f"Processed {len(emails)} email(s)")


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
