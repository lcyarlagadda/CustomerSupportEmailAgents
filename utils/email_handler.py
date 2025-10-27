"""Email handling utilities for Gmail integration."""
import os
import time
import pickle
import base64
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from utils.config import GMAIL_CREDENTIALS_FILE, GMAIL_TOKEN_FILE

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']


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
    
    def check_new_emails(self) -> List[Email]:
        """
        Check for new unread emails from Gmail.
        
        Returns:
            List of new emails
        """
        try:
            results = self.service.users().messages().list(
                userId='me',
                q='is:unread in:inbox',
                maxResults=10
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return []
            
            emails = []
            for msg in messages:
                msg_id = msg['id']
                
                if msg_id in self.processed_ids:
                    continue
                
                message = self.service.users().messages().get(
                    userId='me',
                    id=msg_id,
                    format='full'
                ).execute()
                
                headers = message['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                
                # Get email body
                body = self._get_message_body(message)
                
                # Get thread ID
                thread_id = message.get('threadId', msg_id)
                
                # Parse date
                date_str = next((h['value'] for h in headers if h['name'] == 'Date'), None)
                received_at = datetime.now()  # Simplified, could parse date_str
                
                email = Email(
                    id=msg_id,
                    sender=sender,
                    subject=subject,
                    body=body,
                    received_at=received_at,
                    thread_id=thread_id
                )
                
                emails.append(email)
            
            return emails
            
        except HttpError as error:
            print(f'Error checking emails: {error}')
            return []
    
    def _get_message_body(self, message):
        """Extract body from Gmail message."""
        if 'parts' in message['payload']:
            parts = message['payload']['parts']
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data', '')
                    if data:
                        return base64.urlsafe_b64decode(data).decode('utf-8')
        else:
            data = message['payload']['body'].get('data', '')
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8')
        
        return "No body content"
    
    def mark_as_read(self, email_id: str):
        """
        Mark email as read in Gmail.
        
        Args:
            email_id: ID of email to mark as read
        """
        try:
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            self.processed_ids.add(email_id)
            print(f"Marked email {email_id} as read")
        except HttpError as error:
            print(f'Error marking email as read: {error}')
    
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

