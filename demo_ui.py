"""
Demo UI for Product Support Agents
Simple web interface for testing the system with custom email content.
"""

import streamlit as st
import sys
from io import StringIO
from datetime import datetime

from models.email_model import Email
from workflows.support_workflow import SupportWorkflow


# Page configuration
st.set_page_config(
    page_title="Product Support Agent Demo",
    page_icon="📧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def capture_output(func, *args, **kwargs):
    """Capture stdout from a function."""
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        result = func(*args, **kwargs)
        output = captured_output.getvalue()
        return result, output
    finally:
        sys.stdout = old_stdout


def format_logs(logs):
    """Format logs with color coding."""
    lines = logs.split('\n')
    formatted = []
    
    for line in lines:
        if '[Database]' in line or '[Workflow]' in line:
            formatted.append(f"🔍 {line}")
        elif 'Error' in line or 'error' in line:
            formatted.append(f"❌ {line}")
        elif 'Warning' in line or 'warning' in line:
            formatted.append(f"⚠️ {line}")
        elif line.strip():
            formatted.append(f"   {line}")
    
    return '\n'.join(formatted)


# Header
st.markdown('<div class="main-header">Product Support Agent Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Test the AI-powered email support system with custom email content</div>', unsafe_allow_html=True)

# Initialize workflow
@st.cache_resource
def init_workflow():
    return SupportWorkflow(max_revisions=1, use_cache=False)

with st.spinner("Initializing system..."):
    workflow = init_workflow()

st.success("System ready!")

# Sidebar with sample emails
with st.sidebar:
    st.header("📝 Sample Emails")
    
    sample_emails = {
        "Billing Question": {
            "sender": "john.doe@example.com",
            "subject": "Question about my charge",
            "body": "Hi, I see a charge of $49.99 on my card. Can you tell me what this is for and when my next billing date is?"
        },
        "Technical Support": {
            "sender": "user@company.com",
            "subject": "How do I integrate with Slack?",
            "body": "I want to integrate TaskFlow with our Slack workspace. Can you provide step-by-step instructions?"
        },
        "Account Status": {
            "sender": "john.doe@example.com",
            "subject": "Account information request",
            "body": "Can you tell me what plan I'm on, my current usage, and when my next payment is due?"
        },
        "Feature Request": {
            "sender": "customer@startup.io",
            "subject": "Feature suggestion",
            "body": "It would be great if TaskFlow had a dark mode. This would be really helpful for late-night work sessions."
        }
    }
    
    selected_sample = st.selectbox("Load a sample email:", ["Custom"] + list(sample_emails.keys()))
    
    if st.button("Load Sample", disabled=(selected_sample == "Custom")):
        if selected_sample != "Custom":
            st.session_state.sender = sample_emails[selected_sample]["sender"]
            st.session_state.subject = sample_emails[selected_sample]["subject"]
            st.session_state.body = sample_emails[selected_sample]["body"]
            st.rerun()
    
    st.markdown("---")
    st.markdown("### Database Info")
    st.info("Sample customer: john.doe@example.com\n\nPlan: Pro Plan ($49.99/month)")

# Main input form
st.header("📧 Email Input")

col1, col2 = st.columns([1, 2])

with col1:
    sender = st.text_input(
        "Sender Email",
        value=st.session_state.get("sender", "customer@example.com"),
        placeholder="customer@example.com"
    )

with col2:
    subject = st.text_input(
        "Subject",
        value=st.session_state.get("subject", ""),
        placeholder="Enter email subject..."
    )

body = st.text_area(
    "Email Body",
    value=st.session_state.get("body", ""),
    placeholder="Enter email message...",
    height=200
)

# Process button
if st.button("🚀 Process Email", type="primary", use_container_width=True):
    if not sender or not subject or not body:
        st.error("Please fill in all fields (sender, subject, and body)")
    else:
        # Create email object
        email = Email(
            sender=sender,
            subject=subject,
            body=body,
            thread_id=f"demo_{datetime.now().timestamp()}"
        )
        
        # Process with captured output
        with st.spinner("Processing email..."):
            result, logs = capture_output(workflow.process_email, email)
        
        # Display results
        st.markdown("---")
        st.header("📊 Processing Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔍 System Logs", "✉️ Response", "📈 Details"])
        
        with tab1:
            # Classification
            st.subheader("Email Classification")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Category", result['category'].replace('_', ' ').title())
            with col2:
                st.metric("Priority", result['priority'].upper())
            with col3:
                confidence = result.get('confidence', 0) * 100
                st.metric("Confidence", f"{confidence:.0f}%")
            
            # Database check
            if result.get('rag_sources') and any(s.get('type') == 'database' for s in result['rag_sources']):
                st.markdown('<div class="info-box"><strong>Database Check:</strong> Customer found in database. Using personalized information.</div>', unsafe_allow_html=True)
            
            # QA Results
            st.subheader("Quality Assurance")
            col1, col2 = st.columns(2)
            with col1:
                qa_score = result.get('qa_score', 0)
                st.metric("QA Score", f"{qa_score:.1f}/10")
            with col2:
                approved = "Yes" if result.get('qa_approved') else "No"
                st.metric("Approved", approved)
            
            # Action
            st.subheader("Action Taken")
            action = result.get('action', 'unknown')
            if action == 'send':
                st.markdown('<div class="success-box"><strong>✓ Email will be sent automatically</strong></div>', unsafe_allow_html=True)
            elif action == 'review':
                st.markdown('<div class="warning-box"><strong>⚠ Email flagged for manual review</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info-box"><strong>Action:</strong> {action}</div>', unsafe_allow_html=True)
        
        with tab2:
            st.subheader("System Processing Logs")
            st.code(logs if logs.strip() else "No logs captured", language="text")
        
        with tab3:
            st.subheader("Generated Response")
            response_text = result.get('final_response', 'No response generated')
            
            # Show as email preview
            st.markdown("##### Email Preview")
            st.markdown(f"**To:** {sender}")
            st.markdown(f"**Subject:** Re: {subject}")
            st.markdown("**Body:**")
            st.text_area("", value=response_text, height=300, disabled=True, label_visibility="collapsed")
        
        with tab4:
            st.subheader("Full Processing Details")
            
            # RAG sources
            if result.get('rag_sources'):
                st.markdown("**Information Sources:**")
                for i, source in enumerate(result['rag_sources'], 1):
                    source_type = source.get('type', 'unknown')
                    source_name = source.get('source', 'Unknown')
                    st.markdown(f"{i}. **{source_name}** ({source_type})")
            
            # Issues and suggestions
            if result.get('qa_issues'):
                st.markdown("**QA Issues:**")
                for issue in result['qa_issues']:
                    st.markdown(f"- {issue}")
            
            if result.get('qa_suggestions'):
                st.markdown("**QA Suggestions:**")
                for suggestion in result['qa_suggestions']:
                    st.markdown(f"- {suggestion}")
            
            # Full result (expandable)
            with st.expander("View Raw Result Data"):
                st.json(result)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Product Support Agents - AI-Powered Email Automation</p>
    <p style="font-size: 0.9rem;">Using LangGraph, Groq, and RAG</p>
</div>
""", unsafe_allow_html=True)

