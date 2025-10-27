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

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']


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
        return ''.join(self.text)


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
            "thread_id": self.thread_id
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
            with open(GMAIL_TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("Refreshing expired credentials...")
                creds.refresh(Request())
            else:
                print("Authenticating with Gmail...")
                print(f"Using credentials from: {GMAIL_CREDENTIALS_FILE}")
                flow = InstalledAppFlow.from_client_secrets_file(
                    GMAIL_CREDENTIALS_FILE, SCOPES)
                
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
                    is_colab = 'google.colab' in sys.modules
                    is_headless = is_colab or not os.environ.get('DISPLAY')
                
                if is_headless:
                    print("\n" + "="*60)
                    print("HEADLESS ENVIRONMENT DETECTED (Colab/Server)")
                    print("="*60)
                    print("\nManual authentication required.")
                    print("\nSteps:")
                    print("1. Open the URL below in your browser")
                    print("2. Sign in and authorize the app")
                    print("3. Copy the authorization code from the URL")
                    print("4. Paste it below when prompted")
                    print("="*60 + "\n")
                    
                    # Get authorization URL
                    flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
                    auth_url, _ = flow.authorization_url(prompt='consent')
                    
                    print(f"Authorization URL:\n{auth_url}\n")
                    print("="*60)
                    
                    # Get authorization code from user
                    auth_code = input("\nEnter the authorization code: ").strip()
                    
                    # Exchange code for credentials
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                else:
                    creds = flow.run_local_server(port=0)
            
            with open(GMAIL_TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
            print("Authentication successful")
        
        return build('gmail', 'v1', credentials=creds)
    
    def _create_mock_emails(self) -> List[Email]:
        """Create mock emails for testing."""
        return [
            Email(
                id="email_001",
                sender="john.doe@example.com",
                subject="Can't login to my account",
                body="""Hi TaskFlow Support,

I've been trying to login to my account for the past hour but I keep getting an "Invalid password" error. I'm sure I'm using the correct password. I've tried resetting it twice but I'm not receiving the password reset emails.

This is urgent as I have a project deadline today and need to access my tasks.

Please help!

Best regards,
John Doe""",
                received_at=datetime.now(),
                thread_id="thread_001"
            ),
            Email(
                id="email_002",
                sender="sarah.smith@company.com",
                subject="Question about integrations",
                body="""Hello,

I'm currently on the Professional plan and I'm interested in integrating TaskFlow Pro with our Slack workspace. 

Could you explain how the Slack integration works? Specifically:
1. Can we create tasks directly from Slack messages?
2. Will we get notifications in Slack when tasks are updated?
3. Is there a way to search for tasks within Slack?

Thank you!

Sarah Smith
Project Manager""",
                received_at=datetime.now(),
                thread_id="thread_002"
            ),
            Email(
                id="email_003",
                sender="mike.johnson@startup.io",
                subject="Billing question - upgrading to Enterprise",
                body="""Hi there,

We're currently on the Professional plan with 25 users, and we're considering upgrading to the Enterprise plan.

Can you provide information about:
- Custom pricing for our team size
- What additional features we'd get with Enterprise
- Whether we can pay annually via invoice
- Migration process and timeline

Looking forward to hearing from you.

Mike Johnson
CTO, Startup.io""",
                received_at=datetime.now(),
                thread_id="thread_003"
            ),
            Email(
                id="email_004",
                sender="lisa.wong@design.co",
                subject="Feature request: Dark mode",
                body="""Hey TaskFlow team,

Love your product! We've been using it for 6 months now and it's been great.

One feature request: Could you add a dark mode option? Many of our team members work late hours and would really appreciate having a dark theme to reduce eye strain.

Also, it would be cool to have custom color themes for different projects.

Keep up the great work!

Lisa Wong""",
                received_at=datetime.now(),
                thread_id="thread_004"
            ),
            Email(
                id="email_005",
                sender="spam@marketing.biz",
                subject="Increase your productivity with our amazing tool!",
                body="""Dear Sir/Madam,

We have an amazing productivity tool that will revolutionize your business!!!

Click here to get 50% off NOW!!!

Special limited time offer!!!!

[This is clearly spam and unrelated to TaskFlow Pro support]""",
                received_at=datetime.now(),
                thread_id="thread_005"
            )
        ]
    
    def check_new_emails(self, max_results: int = 10, label: str = 'INBOX') -> List[Email]:
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
            query = f'is:unread label:{label}'
            
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                print("No new emails found.")
                return []
            
            print(f"Found {len(messages)} unread email(s)")
            
            emails = []
            for msg in messages:
                msg_id = msg['id']
                
                # Skip already processed emails
                if msg_id in self.processed_ids:
                    continue
                
                try:
                    # Fetch full message
                    message = self.service.users().messages().get(
                        userId='me',
                        id=msg_id,
                        format='full'
                    ).execute()
                    
                    # Extract headers
                    headers = message['payload']['headers']
                    subject = self._get_header(headers, 'Subject') or 'No Subject'
                    sender_raw = self._get_header(headers, 'From') or 'Unknown'
                    sender = self._extract_email(sender_raw)
                    
                    # Get email body (handles both plain text and HTML)
                    body = self._get_message_body(message)
                    
                    # Get thread ID
                    thread_id = message.get('threadId', msg_id)
                    
                    # Parse date
                    date_str = self._get_header(headers, 'Date')
                    received_at = self._parse_date(date_str) if date_str else datetime.now()
                    
                    email = Email(
                        id=msg_id,
                        sender=sender,
                        subject=subject,
                        body=body,
                        received_at=received_at,
                        thread_id=thread_id
                    )
                    
                    emails.append(email)
                    self.processed_ids.add(msg_id)
                    
                except Exception as e:
                    print(f"Error processing email {msg_id}: {e}")
                    continue
            
            return emails
            
        except HttpError as error:
            print(f'Gmail API error: {error}')
            return []
        except Exception as e:
            print(f'Unexpected error checking emails: {e}')
            return []
    
    def _get_header(self, headers: List[Dict], name: str) -> Optional[str]:
        """Extract header value by name."""
        for header in headers:
            if header['name'].lower() == name.lower():
                return header['value']
        return None
    
    def _extract_email(self, sender_string: str) -> str:
        """
        Extract email address from sender string.
        
        Examples:
            'John Doe <john@example.com>' -> 'john@example.com'
            'john@example.com' -> 'john@example.com'
        """
        email_match = re.search(r'<(.+?)>', sender_string)
        if email_match:
            return email_match.group(1)
        
        # If no angle brackets, check if it's a valid email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', sender_string)
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
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r' +', ' ', text)
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
                return base64.urlsafe_b64decode(data).decode('utf-8')
            except:
                return ""
        
        def extract_from_parts(parts):
            """Recursively extract body from message parts."""
            plain_text = None
            html_text = None
            
            for part in parts:
                mime_type = part.get('mimeType', '')
                
                # Handle nested parts (multipart/alternative, etc.)
                if 'parts' in part:
                    nested_plain, nested_html = extract_from_parts(part['parts'])
                    if nested_plain:
                        plain_text = nested_plain
                    if nested_html:
                        html_text = nested_html
                
                # Extract plain text
                elif mime_type == 'text/plain':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        plain_text = decode_part(data)
                
                # Extract HTML
                elif mime_type == 'text/html':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        html_text = decode_part(data)
            
            return plain_text, html_text
        
        # Check if message has parts (multipart)
        if 'parts' in message['payload']:
            plain_text, html_text = extract_from_parts(message['payload']['parts'])
            
            # Prefer plain text over HTML
            if plain_text:
                return plain_text
            elif html_text:
                return self._strip_html(html_text)
        
        # Single part message
        else:
            mime_type = message['payload'].get('mimeType', '')
            data = message['payload'].get('body', {}).get('data', '')
            
            if data:
                decoded = decode_part(data)
                if decoded:
                    if mime_type == 'text/html':
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
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            self.processed_ids.add(email_id)
            print(f"Marked email {email_id} as read")
            return True
        except HttpError as error:
            print(f'Error marking email as read: {error}')
            return False
    
    def add_label(self, email_id: str, label_name: str) -> bool:
        """
        Add a label to an email.
        
        Args:
            email_id: ID of email
            label_name: Name of label to add
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get or create label
            labels = self.service.users().labels().list(userId='me').execute()
            label_id = None
            
            for label in labels.get('labels', []):
                if label['name'] == label_name:
                    label_id = label['id']
                    break
            
            # Create label if it doesn't exist
            if not label_id:
                label_object = {
                    'name': label_name,
                    'labelListVisibility': 'labelShow',
                    'messageListVisibility': 'show'
                }
                created_label = self.service.users().labels().create(
                    userId='me',
                    body=label_object
                ).execute()
                label_id = created_label['id']
            
            # Add label to email
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'addLabelIds': [label_id]}
            ).execute()
            
            print(f"Added label '{label_name}' to email {email_id}")
            return True
            
        except HttpError as error:
            print(f'Error adding label: {error}')
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
            query = 'is:unread in:inbox -category:promotions -category:social -category:forums -from:noreply'
            
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                print("No new support emails found.")
                return []
            
            print(f"Found {len(messages)} potential support email(s)")
            
            emails = []
            for msg in messages:
                msg_id = msg['id']
                
                if msg_id in self.processed_ids:
                    continue
                
                try:
                    message = self.service.users().messages().get(
                        userId='me',
                        id=msg_id,
                        format='full'
                    ).execute()
                    
                    headers = message['payload']['headers']
                    subject = self._get_header(headers, 'Subject') or 'No Subject'
                    sender_raw = self._get_header(headers, 'From') or 'Unknown'
                    sender = self._extract_email(sender_raw)
                    
                    # Skip obvious automated emails
                    if any(term in sender.lower() for term in ['noreply', 'no-reply', 'donotreply', 'automated']):
                        print(f"Skipping automated email from {sender}")
                        continue
                    
                    body = self._get_message_body(message)
                    thread_id = message.get('threadId', msg_id)
                    date_str = self._get_header(headers, 'Date')
                    received_at = self._parse_date(date_str) if date_str else datetime.now()
                    
                    email = Email(
                        id=msg_id,
                        sender=sender,
                        subject=subject,
                        body=body,
                        received_at=received_at,
                        thread_id=thread_id
                    )
                    
                    emails.append(email)
                    self.processed_ids.add(msg_id)
                    
                except Exception as e:
                    print(f"Error processing email {msg_id}: {e}")
                    continue
            
            return emails
            
        except HttpError as error:
            print(f'Gmail API error: {error}')
            return []
        except Exception as e:
            print(f'Unexpected error: {e}')
            return []
    
    def send_reply(self, email: Email, reply_body: str, subject: str = None):
        """
        Send a reply via Gmail API.
        
        Args:
            email: Original email to reply to
            reply_body: Body of the reply
            subject: Optional subject (defaults to Re: original subject)
        """
        try:
            reply_subject = subject or f"Re: {email.subject}"
            
            # Extract email address from sender (remove name if present)
            to_email = email.sender
            if '<' in to_email:
                to_email = to_email.split('<')[1].split('>')[0]
            
            # Create message
            message = MIMEText(reply_body)
            message['to'] = to_email
            message['subject'] = reply_subject
            message['In-Reply-To'] = email.id
            message['References'] = email.id
            
            # Encode message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            
            print(f"\n" + "=" * 60)
            print("SENDING EMAIL REPLY")
            print("=" * 60)
            print(f"To: {to_email}")
            print(f"Subject: {reply_subject}")
            print(f"Thread ID: {email.thread_id}")
            print("-" * 60)
            print(reply_body[:200] + "..." if len(reply_body) > 200 else reply_body)
            print("=" * 60 + "\n")
            
            # Send message
            sent_message = self.service.users().messages().send(
                userId='me',
                body={
                    'raw': raw_message,
                    'threadId': email.thread_id
                }
            ).execute()
            
            print(f"Email sent successfully (ID: {sent_message['id']})")
            
        except HttpError as error:
            print(f'Error sending email: {error}')
        except Exception as e:
            print(f'Error creating email: {e}')
    
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
        print("\n" + "="*60)
        print("MOCK EMAIL MODE - Using Sample Emails")
        print("="*60)
        print("No Gmail authentication required.")
        print(f"Using {len(self.mock_emails)} pre-generated sample emails.")
        print("="*60 + "\n")
    
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
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                email = Email(
                    id=data.get("id", f"email_{len(emails)+1:03d}"),
                    sender=data["sender"],
                    subject=data["subject"],
                    body=data["body"],
                    received_at=datetime.now()
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
                received_at=datetime(2024, 1, 15, 9, 30, 0)
            )
        ]
    
    def check_new_emails(self) -> List[Email]:
        """Return mock emails (only once per run)."""
        if not self.email_returned:
            self.email_returned = True
            return self.mock_emails
        return []
    
    def send_email(self, to: str, subject: str, body: str, reply_to_id: Optional[str] = None) -> bool:
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
        if '<' in to_email:
            to_email = to_email.split('<')[1].split('>')[0]
        
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


if __name__ == "__main__":
    # Test the email handler
    print("Testing Email Handler...")
    print("-" * 60)
    
    handler = GmailHandler("support@taskflowpro.com")
    
    # Check for new emails
    new_emails = handler.check_new_emails()
    print(f"Found {len(new_emails)} new emails:\n")
    
    for email in new_emails:
        print(f"From: {email.sender}")
        print(f"Subject: {email.subject}")
        print(f"Preview: {email.body[:100]}...")
        print("-" * 60)

