"""Email handling utilities for Gmail integration."""

import os
import time
import pickle
import base64
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from email.mime.text import MIMEText
from html import unescape
from html.parser import HTMLParser

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from utils.config import GMAIL_CREDENTIALS_FILE, GMAIL_TOKEN_FILE

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


class HTMLStripper(HTMLParser):
    """Strip HTML tags from text."""

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, d):
        self.text.append(d)

    def get_data(self):
        return "".join(self.text)


@dataclass
class Email:
    """Represents an email message."""

    id: str
    sender: str
    subject: str
    body: str
    received_at: datetime
    thread_id: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert email to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "subject": self.subject,
            "body": self.body,
            "received_at": self.received_at.isoformat(),
            "thread_id": self.thread_id,
        }


class GmailHandler:
    """Handler for Gmail API operations using official Google API."""

    def __init__(self, email_address: str):
        """
        Initialize Gmail handler.

        Args:
            email_address: Gmail address to monitor
        """
        self.email_address = email_address
        self.service = self._authenticate()
        self.processed_ids = set()

    def _authenticate(self):
        """Authenticate with Gmail API using OAuth2."""
        creds = None

        if os.path.exists(GMAIL_TOKEN_FILE):
            with open(GMAIL_TOKEN_FILE, "rb") as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("Refreshing expired credentials...")
                creds.refresh(Request())
            else:
                print("Authenticating with Gmail...")
                print(f"Using credentials from: {GMAIL_CREDENTIALS_FILE}")
                flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, SCOPES)

                # Check if running in headless environment (Colab, server, etc.)
                try:
                    import webbrowser

                    webbrowser.get()
                    is_headless = False
                except:
                    is_headless = True

                # Also check for common headless indicators
                if not is_headless:
                    import sys

                    is_colab = "google.colab" in sys.modules
                    is_headless = is_colab or not os.environ.get("DISPLAY")

                if is_headless:
                    print("\n" + "=" * 60)
                    print("HEADLESS ENVIRONMENT DETECTED (Colab/Server)")
                    print("=" * 60)
                    print("\nManual authentication required.")
                    print("\nSteps:")
                    print("1. Open the URL below in your browser")
                    print("2. Sign in and authorize the app")
                    print("3. Copy the authorization code from the URL")
                    print("4. Paste it below when prompted")
                    print("=" * 60 + "\n")

                    # Get authorization URL
                    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
                    auth_url, _ = flow.authorization_url(prompt="consent")

                    print(f"Authorization URL:\n{auth_url}\n")
                    print("=" * 60)

                    # Get authorization code from user
                    auth_code = input("\nEnter the authorization code: ").strip()

                    # Exchange code for credentials
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                else:
                    creds = flow.run_local_server(port=0)

            with open(GMAIL_TOKEN_FILE, "wb") as token:
                pickle.dump(creds, token)
            print("Authentication successful")

        return build("gmail", "v1", credentials=creds)

    def check_new_emails(self, max_results: int = 10, label: str = "INBOX") -> List[Email]:
        """
        Check for new unread emails from Gmail.

        Args:
            max_results: Maximum number of emails to fetch
            label: Gmail label to filter by (default: INBOX)

        Returns:
            List of new emails
        """
        try:
            # Query for unread emails in specified label
            query = f"is:unread label:{label}"

            results = (
                self.service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )

            messages = results.get("messages", [])

            if not messages:
                print("No new emails found.")
                return []

            print(f"Found {len(messages)} unread email(s)")

            emails = []
            for msg in messages:
                msg_id = msg["id"]

                # Skip already processed emails
                if msg_id in self.processed_ids:
                    continue

                try:
                    # Fetch full message
                    message = (
                        self.service.users()
                        .messages()
                        .get(userId="me", id=msg_id, format="full")
                        .execute()
                    )

                    # Extract headers
                    headers = message["payload"]["headers"]
                    subject = self._get_header(headers, "Subject") or "No Subject"
                    sender_raw = self._get_header(headers, "From") or "Unknown"
                    sender = self._extract_email(sender_raw)

                    # Get email body (handles both plain text and HTML)
                    body = self._get_message_body(message)

                    # Get thread ID
                    thread_id = message.get("threadId", msg_id)

                    # Parse date
                    date_str = self._get_header(headers, "Date")
                    received_at = self._parse_date(date_str) if date_str else datetime.now()

                    email = Email(
                        id=msg_id,
                        sender=sender,
                        subject=subject,
                        body=body,
                        received_at=received_at,
                        thread_id=thread_id,
                    )

                    emails.append(email)
                    self.processed_ids.add(msg_id)

                except Exception as e:
                    print(f"Error processing email {msg_id}: {e}")
                    continue

            return emails

        except HttpError as error:
            print(f"Gmail API error: {error}")
            return []
        except Exception as e:
            print(f"Unexpected error checking emails: {e}")
            return []

    def _get_header(self, headers: List[Dict], name: str) -> Optional[str]:
        """Extract header value by name."""
        for header in headers:
            if header["name"].lower() == name.lower():
                return header["value"]
        return None

    def _extract_email(self, sender_string: str) -> str:
        """
        Extract email address from sender string.

        Examples:
            'John Doe <john@example.com>' -> 'john@example.com'
            'john@example.com' -> 'john@example.com'
        """
        email_match = re.search(r"<(.+?)>", sender_string)
        if email_match:
            return email_match.group(1)

        # If no angle brackets, check if it's a valid email
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", sender_string)
        if email_match:
            return email_match.group(0)

        return sender_string

    def _parse_date(self, date_str: str) -> datetime:
        """Parse email date string to datetime."""
        try:
            from email.utils import parsedate_to_datetime

            return parsedate_to_datetime(date_str)
        except:
            return datetime.now()

    def _strip_html(self, html_content: str) -> str:
        """Strip HTML tags and return plain text."""
        stripper = HTMLStripper()
        try:
            stripper.feed(html_content)
            text = stripper.get_data()
            # Clean up extra whitespace
            text = re.sub(r"\n\s*\n", "\n\n", text)
            text = re.sub(r" +", " ", text)
            return text.strip()
        except:
            return html_content

    def _get_message_body(self, message):
        """
        Extract body from Gmail message, handling multipart and HTML content.

        Priority:
        1. Plain text (text/plain)
        2. HTML converted to plain text (text/html)
        3. Fallback message
        """

        def decode_part(data: str) -> str:
            """Decode base64 encoded part."""
            try:
                return base64.urlsafe_b64decode(data).decode("utf-8")
            except:
                return ""

        def extract_from_parts(parts):
            """Recursively extract body from message parts."""
            plain_text = None
            html_text = None

            for part in parts:
                mime_type = part.get("mimeType", "")

                # Handle nested parts (multipart/alternative, etc.)
                if "parts" in part:
                    nested_plain, nested_html = extract_from_parts(part["parts"])
                    if nested_plain:
                        plain_text = nested_plain
                    if nested_html:
                        html_text = nested_html

                # Extract plain text
                elif mime_type == "text/plain":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        plain_text = decode_part(data)

                # Extract HTML
                elif mime_type == "text/html":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        html_text = decode_part(data)

            return plain_text, html_text

        # Check if message has parts (multipart)
        if "parts" in message["payload"]:
            plain_text, html_text = extract_from_parts(message["payload"]["parts"])

            # Prefer plain text over HTML
            if plain_text:
                return plain_text
            elif html_text:
                return self._strip_html(html_text)

        # Single part message
        else:
            mime_type = message["payload"].get("mimeType", "")
            data = message["payload"].get("body", {}).get("data", "")

            if data:
                decoded = decode_part(data)
                if decoded:
                    if mime_type == "text/html":
                        return self._strip_html(decoded)
                    return decoded

        return "No body content"

    def mark_as_read(self, email_id: str) -> bool:
        """
        Mark email as read in Gmail.

        Args:
            email_id: ID of email to mark as read

        Returns:
            True if successful, False otherwise
        """
        try:
            self.service.users().messages().modify(
                userId="me", id=email_id, body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            self.processed_ids.add(email_id)
            print(f"Marked email {email_id} as read")
            return True
        except HttpError as error:
            print(f"Error marking email as read: {error}")
            return False

    def add_label(self, email_id: str, label_name: str) -> bool:
        """
        Add a label to an email and verify it was applied.

        Args:
            email_id: ID of email
            label_name: Name of label to add

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get or create label
            labels = self.service.users().labels().list(userId="me").execute()
            label_id = None

            for label in labels.get("labels", []):
                if label["name"] == label_name:
                    label_id = label["id"]
                    break

            # Create label if it doesn't exist
            if not label_id:
                print(f"Label '{label_name}' not found. Creating new label...")
                label_object = {
                    "name": label_name,
                    "labelListVisibility": "labelShow",  # Show in label list
                    "messageListVisibility": "show",  # Show in message list
                }
                created_label = (
                    self.service.users().labels().create(userId="me", body=label_object).execute()
                )
                label_id = created_label["id"]
                print(f"Created label '{label_name}' (ID: {label_id})")

            # Add label to email (also keep it unread for visibility)
            print(f"Applying label to email {email_id}...")
            modified_message = (
                self.service.users()
                .messages()
                .modify(
                    userId="me",
                    id=email_id,
                    body={
                        "addLabelIds": [label_id],
                        "removeLabelIds": [],  # Explicitly keep UNREAD label
                    },
                )
                .execute()
            )

            # Verify label was applied
            applied_labels = modified_message.get("labelIds", [])
            label_names = []
            for lbl in labels.get("labels", []):
                if lbl["id"] in applied_labels:
                    label_names.append(lbl["name"])

            if label_id in applied_labels:
                print(f"{'=' * 70}")
                print(f"SUCCESS: Label '{label_name}' applied to email {email_id}")
                print(f"Email now has labels: {', '.join(label_names) if label_names else 'None'}")
                print(f"{'=' * 70}")
                print(f"\nTo view labeled emails in Gmail:")
                print(f"1. In Gmail, search for: label:{label_name} is:unread")
                print(f"2. Or click the '{label_name}' label in the left sidebar")
                print(f"3. Check Settings > Labels to ensure label is visible\n")
                return True
            else:
                print(f"{'=' * 70}")
                print(f"WARNING: Label '{label_name}' may not have been applied correctly")
                print(f"Expected label ID: {label_id}")
                print(f"Applied label IDs: {applied_labels}")
                print(f"{'=' * 70}\n")
                return False

        except HttpError as error:
            error_content = (
                error.content.decode("utf-8") if hasattr(error, "content") else str(error)
            )
            print(f"\n{'=' * 70}")
            print(f"ERROR: Failed to add label '{label_name}' to email {email_id}")
            print(f"HTTP Status: {error.resp.status if hasattr(error, 'resp') else 'Unknown'}")
            print(f"Error Details: {error_content}")
            print(f"{'=' * 70}\n")
            return False
        except Exception as e:
            import traceback

            print(f"\n{'=' * 70}")
            print(f"ERROR: Unexpected error adding label '{label_name}' to email {email_id}")
            print(f"Error: {e}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Traceback:\n{traceback.format_exc()}")
            print(f"{'=' * 70}\n")
            return False

    def check_new_support_emails(self, max_results: int = 10) -> List[Email]:
        """
        Check for new support-related emails.
        Filters out spam, promotions, and social emails.

        Args:
            max_results: Maximum number of emails to fetch

        Returns:
            List of support emails
        """
        try:
            # Query that filters out common non-support categories
            query = "is:unread in:inbox -category:promotions -category:social -category:forums -from:noreply"

            results = (
                self.service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )

            messages = results.get("messages", [])

            if not messages:
                print("No new support emails found.")
                return []

            print(f"Found {len(messages)} potential support email(s)")

            emails = []
            for msg in messages:
                msg_id = msg["id"]

                if msg_id in self.processed_ids:
                    continue

                try:
                    message = (
                        self.service.users()
                        .messages()
                        .get(userId="me", id=msg_id, format="full")
                        .execute()
                    )

                    headers = message["payload"]["headers"]
                    subject = self._get_header(headers, "Subject") or "No Subject"
                    sender_raw = self._get_header(headers, "From") or "Unknown"
                    sender = self._extract_email(sender_raw)

                    # Skip obvious automated emails
                    if any(
                        term in sender.lower()
                        for term in ["noreply", "no-reply", "donotreply", "automated"]
                    ):
                        print(f"Skipping automated email from {sender}")
                        continue

                    body = self._get_message_body(message)
                    thread_id = message.get("threadId", msg_id)
                    date_str = self._get_header(headers, "Date")
                    received_at = self._parse_date(date_str) if date_str else datetime.now()

                    email = Email(
                        id=msg_id,
                        sender=sender,
                        subject=subject,
                        body=body,
                        received_at=received_at,
                        thread_id=thread_id,
                    )

                    emails.append(email)
                    self.processed_ids.add(msg_id)

                except Exception as e:
                    print(f"Error processing email {msg_id}: {e}")
                    continue

            return emails

        except HttpError as error:
            print(f"Gmail API error: {error}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def _normalize_message_id(self, message_id: str) -> str:
        """
        Normalize Message-ID to ensure proper formatting.

        Args:
            message_id: Message-ID string (may have whitespace or missing brackets)

        Returns:
            Normalized Message-ID in angle brackets
        """
        if not message_id:
            return ""
        # Strip whitespace
        message_id = message_id.strip()
        # Remove angle brackets if present (we'll add them)
        message_id = message_id.strip("<>")
        # Ensure it's in angle brackets
        return f"<{message_id}>"

    def _build_references_header(self, original_headers: List[Dict]) -> str:
        """
        Build proper References header from original email headers.

        Args:
            original_headers: List of header dictionaries from original message

        Returns:
            Properly formatted References header value
        """
        # Get existing References
        references_str = self._get_header(original_headers, "References") or ""
        # Get In-Reply-To
        in_reply_to = self._get_header(original_headers, "In-Reply-To") or ""
        # Get Message-ID
        message_id = self._get_header(original_headers, "Message-ID") or ""

        # Normalize Message-ID
        normalized_msg_id = self._normalize_message_id(message_id)

        # Build reference list
        ref_list = []

        # Add existing References (split by spaces and normalize each)
        if references_str:
            # Split references and normalize each
            existing_refs = re.split(r"\s+", references_str.strip())
            for ref in existing_refs:
                if ref:
                    normalized_ref = self._normalize_message_id(ref)
                    if normalized_ref and normalized_ref not in ref_list:
                        ref_list.append(normalized_ref)

        # Add In-Reply-To if not already in References
        if in_reply_to:
            normalized_in_reply_to = self._normalize_message_id(in_reply_to)
            if normalized_in_reply_to and normalized_in_reply_to not in ref_list:
                ref_list.append(normalized_in_reply_to)

        # Add current Message-ID if not already present
        if normalized_msg_id and normalized_msg_id not in ref_list:
            ref_list.append(normalized_msg_id)

        # Return space-separated string
        return " ".join(ref_list) if ref_list else normalized_msg_id

    def send_reply(self, email: Email, reply_body: str, subject: str = None):
        """
        Send a reply via Gmail API as part of the original thread.

        Args:
            email: Original email to reply to
            reply_body: Body of the reply (should be properly formatted)
            subject: Optional subject (defaults to Re: original subject)
        """
        try:
            # Handle subject line - avoid duplicate "Re:" prefixes
            if subject:
                reply_subject = subject
            elif email.subject.lower().startswith("re:"):
                reply_subject = email.subject
            else:
                reply_subject = f"Re: {email.subject}"

            # Extract email address from sender (remove name if present)
            to_email = email.sender
            if "<" in to_email:
                to_email = to_email.split("<")[1].split(">")[0]

            # Get the authenticated user's email address for From header
            profile = self.service.users().getProfile(userId="me").execute()
            from_email = profile.get("emailAddress", self.email_address)

            # Get the original message to retrieve Message-ID header for proper threading
            original_message = (
                self.service.users()
                .messages()
                .get(userId="me", id=email.id, format="full")
                .execute()
            )
            headers = original_message["payload"]["headers"]

            # Extract and normalize Message-ID from original email
            message_id_raw = self._get_header(headers, "Message-ID")
            if not message_id_raw:
                # Fallback: create a Message-ID if not present
                message_id_raw = f"{email.id}@mail.gmail.com"
            message_id = self._normalize_message_id(message_id_raw)

            # Build proper References header
            references = self._build_references_header(headers)

            # Normalize reply body - ensure perfect formatting
            # Remove all leading whitespace (spaces, tabs) from each line
            lines = reply_body.split("\n")
            cleaned_lines = []
            for line in lines:
                # Remove ALL leading whitespace (spaces, tabs, non-breaking spaces)
                cleaned_line = line.lstrip(" \t\u00a0").rstrip(" \t")
                cleaned_lines.append(cleaned_line)

            # Join and normalize multiple blank lines
            reply_body = "\n".join(cleaned_lines)
            # Remove leading/trailing whitespace from entire body
            reply_body = reply_body.strip()
            # Normalize paragraph spacing (max 2 consecutive newlines)
            reply_body = re.sub(r"\n{3,}", "\n\n", reply_body)

            # Create message with proper threading headers
            message = MIMEText(reply_body, "plain", "utf-8")
            message["From"] = from_email
            message["To"] = to_email
            message["Subject"] = reply_subject
            message["In-Reply-To"] = message_id
            message["References"] = references

            # Encode message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

            # Send as reply in thread (use threadId for proper threading)
            sent_message = (
                self.service.users()
                .messages()
                .send(userId="me", body={"raw": raw_message, "threadId": email.thread_id})
                .execute()
            )

            print(f"Email sent successfully (ID: {sent_message['id']})")

        except HttpError as error:
            print(f"Error sending email: {error}")
        except Exception as e:
            print(f"Error creating email: {e}")

    def get_email_by_id(self, email_id: str) -> Email:
        """
        Retrieve a specific email by ID.

        Args:
            email_id: Email ID

        Returns:
            Email object or None
        """
        for email in self.mock_emails:
            if email.id == email_id:
                return email
        return None


# For production use with actual Gmail API:
"""
def create_gmail_service():
    '''Create Gmail API service with authentication.'''
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import pickle
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)
"""


class MockEmailHandler:
    """Mock email handler that returns sample emails for testing/demo purposes."""

    def __init__(self, email_address: str):
        """
        Initialize mock handler.

        Args:
            email_address: Email address (not used, for compatibility)
        """
        self.email_address = email_address
        self.mock_emails = self._create_mock_emails()
        self.email_returned = False
        print("\n" + "=" * 60)
        print("MOCK EMAIL MODE - Using Sample Emails")
        print("=" * 60)
        print("No Gmail authentication required.")
        print(f"Using {len(self.mock_emails)} pre-generated sample emails.")
        print("=" * 60 + "\n")

    def _create_mock_emails(self) -> List[Email]:
        """Load mock emails from JSON files."""
        import json
        from pathlib import Path
        from datetime import datetime

        test_emails_dir = Path(__file__).parent.parent / "data" / "test_emails"

        if not test_emails_dir.exists():
            print(f"Warning: Test emails directory not found at {test_emails_dir}")
            return self._get_fallback_emails()

        emails = []
        json_files = sorted(test_emails_dir.glob("*.json"))

        if not json_files:
            print(f"Warning: No JSON files found in {test_emails_dir}")
            return self._get_fallback_emails()

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                email = Email(
                    id=data.get("id", f"email_{len(emails)+1:03d}"),
                    sender=data["sender"],
                    subject=data["subject"],
                    body=data["body"],
                    received_at=datetime.now(),
                )
                emails.append(email)
            except Exception as e:
                print(f"Warning: Failed to load {json_file.name}: {e}")
                continue

        return emails if emails else self._get_fallback_emails()

    def _get_fallback_emails(self) -> List[Email]:
        """Fallback emails if JSON files can't be loaded."""
        from datetime import datetime

        return [
            Email(
                id="email_001",
                sender="john.doe@example.com",
                subject="Can't login to my account",
                body="Hi TaskFlow Support,\n\nI'm having trouble logging into my account. Can you help?\n\nThanks,\nJohn",
                received_at=datetime(2024, 1, 15, 9, 30, 0),
            )
        ]

    def check_new_emails(self) -> List[Email]:
        """Return mock emails (only once per run)."""
        if not self.email_returned:
            self.email_returned = True
            return self.mock_emails
        return []

    def send_email(
        self, to: str, subject: str, body: str, reply_to_id: Optional[str] = None
    ) -> bool:
        """
        Simulate sending an email.

        Args:
            to: Recipient email
            subject: Email subject
            body: Email body
            reply_to_id: ID of email being replied to

        Returns:
            Always True (simulated success)
        """
        print(f"\n[MOCK] Would send email to: {to}")
        print(f"[MOCK] Subject: {subject}")
        print(f"[MOCK] Body preview: {body[:100]}...")
        print("[MOCK] (Email not actually sent in mock mode)\n")
        return True

    def send_reply(self, email: Email, reply_body: str, subject: str = None):
        """
        Simulate sending a reply email.

        Args:
            email: Original email to reply to
            reply_body: Body of the reply
            subject: Optional subject (defaults to Re: original subject)
        """
        reply_subject = subject or f"Re: {email.subject}"

        # Extract email address from sender
        to_email = email.sender
        if "<" in to_email:
            to_email = to_email.split("<")[1].split(">")[0]

        print(f"\n" + "=" * 60)
        print("[MOCK] SENDING EMAIL REPLY")
        print("=" * 60)
        print(f"To: {to_email}")
        print(f"Subject: {reply_subject}")
        print(f"Thread ID: {getattr(email, 'thread_id', email.id)}")
        print("-" * 60)
        print(reply_body[:200] + "..." if len(reply_body) > 200 else reply_body)
        print("=" * 60)
        print("[MOCK] (Email not actually sent in mock mode)\n")

    def mark_as_read(self, email_id: str) -> bool:
        """
        Simulate marking email as read.

        Args:
            email_id: Email ID

        Returns:
            Always True (simulated success)
        """
        print(f"[MOCK] Marked email {email_id} as read")
        return True

    def add_label(self, email_id: str, label_name: str) -> bool:
        """
        Simulate adding a label to an email.

        Args:
            email_id: Email ID
            label_name: Name of label to add

        Returns:
            Always True (simulated success)
        """
        print(f"[MOCK] Added label '{label_name}' to email {email_id}")
        return True
