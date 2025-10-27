"""Response generation agent for crafting email replies."""
from typing import Dict, Any, Optional
from langchain_community.llms import HuggingFacePipeline

from utils.llm_loader import load_llm_pipeline


class ResponseGeneratorAgent:
    """Agent responsible for generating email responses."""
    
    def __init__(self):
        """Initialize the response generator agent."""
        pipe = load_llm_pipeline(temperature=0.8)
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Prompts for different email categories (string templates)
        self.prompts = {
            "technical_support": """Customer Email - From: {sender}, Subject: {subject}
Body: {body}

Context: {context}

Write ONLY an email response (200-400 words) that:
- Shows empathy and acknowledges frustration
- Provides 3-5 clear troubleshooting steps
- Offers 2-3 solutions
- Ends with reassurance

Start with "Dear" or "Hi". Output ONLY the email, no explanations.""",
            
            "product_inquiry": """Customer Email - From: {sender}, Subject: {subject}
Body: {body}

Context: {context}

Write ONLY an email response (200-350 words) that:
- Answers questions directly and accurately
- Uses bullet points for features
- Suggests next steps

Start with "Dear" or "Hi". Output ONLY the email, no explanations.""",
            
            "billing": """Customer Email - From: {sender}, Subject: {subject}
Body: {body}

Context: {context}

Write ONLY an email response (150-300 words) that:
- Is clear and transparent
- Shows empathy
- Provides specific next steps

Start with "Dear" or "Hi". Output ONLY the email, no explanations.""",
            
            "feature_request": """Customer Email - From: {sender}, Subject: {subject}
Body: {body}

Context: {context}

Write ONLY an email response (150-250 words) that:
- Thanks them sincerely
- Explains if similar features exist
- Suggests workarounds if available

Start with "Dear" or "Hi". Output ONLY the email, no explanations.""",
            
            "feedback": """Customer Email - From: {sender}, Subject: {subject}
Body: {body}

Context: {context}

Write ONLY an email response (100-200 words) that:
- Thanks them for feedback
- Shows appreciation (positive) or apologizes (negative)
- Invites ongoing communication

Start with "Dear" or "Hi". Output ONLY the email, no explanations."""
        }
    
    def generate_response(
        self,
        email_data: Dict[str, Any],
        category: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate an email response.
        
        Args:
            email_data: Dictionary with email information
                - sender: Sender's email address
                - subject: Email subject
                - body: Email body content
            category: Email category (technical_support, product_inquiry, etc.)
            context: Optional additional context (e.g., from RAG agent)
        
        Returns:
            Generated email response
        """
        # Get appropriate prompt for category
        prompt = self.prompts.get(category)
        
        if not prompt:
            # Fallback to general response
            prompt = self._get_general_prompt()
        
        # Prepare context
        if context:
            context_text = context
        else:
            context_text = "No additional context provided. Use your general knowledge of TaskFlow Pro."
        
        # Format prompt with data
        formatted_prompt = prompt.format(
            sender=email_data.get("sender", "Valued Customer"),
            subject=email_data.get("subject", "Your inquiry"),
            body=email_data.get("body", ""),
            context=context_text
        )
        
        # Generate response
        raw_response = self.llm.invoke(formatted_prompt)
        
        # Clean up the response (remove prompt echo)
        response = self._clean_response(raw_response, email_data)
        
        return response
    
    def _clean_response(self, raw_response: str, email_data: dict) -> str:
        """
        Clean up LLM response by removing prompt echo and extracting actual email.
        
        Args:
            raw_response: Raw LLM output
            email_data: Original email data
        
        Returns:
            Clean email response
        """
        # Common markers that indicate the start of the actual response
        start_markers = [
            "Dear ",
            "Hi ",
            "Hello ",
            "Thank you for",
            "Thanks for",
            f"Dear {email_data.get('sender', '').split('@')[0]}",
        ]
        
        # Try to find where the actual response starts
        response = raw_response
        
        # Remove everything before the first greeting if prompt is echoed
        for marker in start_markers:
            if marker in response:
                # Find the first occurrence of a greeting
                idx = response.find(marker)
                if idx > 100:  # Only trim if there's significant text before
                    response = response[idx:]
                break
        
        # Remove any trailing meta-commentary (text after signature)
        signature_markers = [
            "Best regards,",
            "Sincerely,",
            "Best,",
            "Regards,",
            "TaskFlow Support Team",
            "TaskFlow Pro Support"
        ]
        
        # Find the last signature and cut after it
        last_sig_pos = -1
        last_sig_len = 0
        for marker in signature_markers:
            pos = response.rfind(marker)
            if pos > last_sig_pos:
                last_sig_pos = pos
                last_sig_len = len(marker)
        
        if last_sig_pos > 0:
            # Keep text up to 100 chars after the signature (for closing lines)
            end_pos = last_sig_pos + last_sig_len + 100
            potential_end = response[last_sig_pos:end_pos]
            
            # Find where the meta-commentary starts
            meta_markers = [
                "\n\nThis response",
                "\n\nThe response",
                "\n\nNote:",
                "\n\nI hope",
                "\n\nPlease note",
                "Use the relevant information"
            ]
            
            for meta_marker in meta_markers:
                if meta_marker in potential_end:
                    meta_pos = potential_end.find(meta_marker)
                    response = response[:last_sig_pos + meta_pos]
                    break
            else:
                # No meta-commentary found, keep reasonable amount after signature
                lines_after = response[last_sig_pos + last_sig_len:].split('\n')
                # Keep at most 3 lines after signature
                if len(lines_after) > 3:
                    response = response[:last_sig_pos + last_sig_len] + '\n'.join(lines_after[:3])
        
        # Remove any remaining prompt artifacts
        prompt_artifacts = [
            "You are a skilled technical support agent",
            "Your task is to write",
            "Guidelines:",
            "Context/Information to use:",
            "Original Email:",
            "Write a complete but CONCISE",
            "Use the relevant information from"
        ]
        
        for artifact in prompt_artifacts:
            if artifact in response[:200]:  # Only check beginning
                # This is likely still the prompt, try harder to find actual response
                lines = response.split('\n')
                for i, line in enumerate(lines):
                    if any(marker in line for marker in start_markers):
                        response = '\n'.join(lines[i:])
                        break
        
        return response.strip()
    
    def _get_general_prompt(self) -> str:
        """Get a general-purpose response prompt."""
        return """Customer Email - From: {sender}, Subject: {subject}
Body: {body}

Context: {context}

Write ONLY an email response (200-400 words) that:
- Is professional and helpful
- Addresses their concerns directly
- Provides next steps
- Shows empathy

Start with "Dear" or "Hi". Output ONLY the email, no explanations."""
    
    def add_signature(self, email_body: str, agent_name: str = "TaskFlow Support Team") -> str:
        """
        Add a professional signature to the email.
        
        Args:
            email_body: The email body content
            agent_name: Name to sign with
        
        Returns:
            Email with signature
        """
        signature = f"""

Best regards,
{agent_name}

---
TaskFlow Pro Support
support@taskflowpro.com
https://help.taskflowpro.com

Need immediate help? Chat with us at app.taskflowpro.com (Mon-Fri, 9am-6pm EST)"""
        
        return email_body + signature


if __name__ == "__main__":
    # Test the response generator
    print("Testing Response Generator Agent...")
    print("=" * 60)
    
    agent = ResponseGeneratorAgent()
    
    # Test email
    test_email = {
        "sender": "john.doe@example.com",
        "subject": "Can't access my account",
        "body": """Hi,

I've been trying to log in for the past hour but keep getting an error.
I'm sure my password is correct. This is urgent!

John"""
    }
    
    test_context = """Password reset emails are sent from support@taskflowpro.com.
They may take 5-10 minutes to arrive.
Check spam folder if not received.
Password reset links expire after 24 hours."""
    
    print("\nGenerating response for technical support email...")
    print("-" * 60)
    
    try:
        response = agent.generate_response(
            email_data=test_email,
            category="technical_support",
            context=test_context
        )
        
        # Add signature
        final_response = agent.add_signature(response)
        
        print("\nGenerated Response:")
        print("=" * 60)
        print(final_response)
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

