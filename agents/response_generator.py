"""Response generation agent for crafting email replies."""
from typing import Dict, Any, Optional

from utils.unified_llm_loader import load_llm


class ResponseGeneratorAgent:
    """Agent responsible for generating email responses."""
    
    def __init__(self):
        """Initialize the response generator agent."""
        # Response generation needs longer outputs - use 400 tokens (reduced from 512)
        self.llm = load_llm(temperature=0.8, max_tokens=400)
        
        # Prompts for different email categories (string templates)
        self.prompts = {
            "technical_support": """Customer Email: {subject}
{body}

Available Info: {context}

Write ONLY the email body (200-400 words). Start with greeting, end BEFORE signature. 
DO NOT include: [placeholders], notes, explanations, signature, or "please let me know".""",
            
            "product_inquiry": """Customer Email: {subject}
{body}

Available Info: {context}

Write ONLY the email body (200-350 words). Start with greeting, end BEFORE signature.
DO NOT include: [placeholders], notes, explanations, signature, or "please let me know".""",
            
            "billing": """Customer Email: {subject}
{body}

Available Info: {context}

Write ONLY the email body (150-300 words). Start with greeting, end BEFORE signature.
DO NOT include: [placeholders], notes, explanations, signature, or "please let me know".""",
            
            "feature_request": """Customer Email: {subject}
{body}

Available Info: {context}

Write ONLY the email body (150-250 words). Start with greeting, end BEFORE signature.
DO NOT include: [placeholders], notes, explanations, signature, or "please let me know".""",
            
            "feedback": """Customer Email: {subject}
{body}

Available Info: {context}

Write ONLY the email body (100-200 words). Start with greeting, end BEFORE signature.
DO NOT include: [placeholders], notes, explanations, signature, or "please let me know"."""
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
        
        # Handle different response types (Groq returns AIMessage, HuggingFace returns string)
        if hasattr(raw_response, 'content'):
            # Groq/ChatGPT style AIMessage
            response_text = raw_response.content
        else:
            # HuggingFace style string
            response_text = raw_response
        
        # Clean up the response (remove prompt echo)
        response = self._clean_response(response_text, email_data)
        
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
        
        # Find the FIRST signature and cut there (we'll add our own later)
        signature_markers = [
            "Best regards,",
            "Sincerely,",
            "Best,",
            "Warm regards,",
            "Kind regards,",
            "Regards,",
            "Thank you,",
        ]
        
        # Find the FIRST signature position
        first_sig_pos = len(response)
        for marker in signature_markers:
            pos = response.find(marker)
            if pos > 0 and pos < first_sig_pos:
                first_sig_pos = pos
        
        # Cut at first signature
        if first_sig_pos < len(response):
            response = response[:first_sig_pos].rstrip()
        
        # Remove placeholder text (like [Your Name], [Phone], etc.)
        import re
        response = re.sub(r'\[Your [^\]]+\]', '', response)
        response = re.sub(r'\[Your [^\]]+\]\s*\(optional\)', '', response)
        response = re.sub(r'\(optional\)', '', response)
        
        # Remove meta-commentary and notes
        meta_patterns = [
            r'\n\nNote:.*$',
            r'\n\nPlease note.*$',
            r'\n\nPlease let me know.*$',
            r'\n\nThis response.*$',
            r'\n\nThe response.*$',
            r'\n\nI hope this.*$',
            r'\n\nFeel free to.*$',
            r'\n\nIf you need any modifications.*$',
            r'\n\n---.*$',  # Remove trailing separators
        ]
        
        for pattern in meta_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any remaining prompt artifacts
        prompt_artifacts = [
            "You are a skilled technical support agent",
            "Your task is to write",
            "Guidelines:",
            "Context/Information to use:",
            "Original Email:",
            "Write a complete but CONCISE",
            "Use the relevant information from",
            "Customer Email -"
        ]
        
        for artifact in prompt_artifacts:
            if artifact in response[:200]:  # Only check beginning
                # This is likely still the prompt, try harder to find actual response
                lines = response.split('\n')
                for i, line in enumerate(lines):
                    if any(marker in line for marker in start_markers):
                        response = '\n'.join(lines[i:])
                        break
        
        # Clean up extra whitespace
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:  # Keep non-empty lines
                cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines)
        
        return response.strip()
    
    def _get_general_prompt(self) -> str:
        """Get a general-purpose response prompt."""
        return """Customer Email: {subject}
{body}

Available Info: {context}

Write ONLY the email body (200-400 words). Start with greeting, end BEFORE signature.
DO NOT include: [placeholders], notes, explanations, signature, or "please let me know"."""
    
    def add_signature(self, email_body: str, agent_name: str = "TaskFlow Support Team") -> str:
        """
        Add a professional signature to the email.
        
        Args:
            email_body: The email body content (should NOT already have a signature)
            agent_name: Name to sign with
        
        Returns:
            Email with signature
        """
        # Ensure body doesn't already have a signature
        signature_check = ["Best regards", "Sincerely", "TaskFlow Support"]
        if any(sig in email_body for sig in signature_check):
            # Already has signature, don't add another
            return email_body
        
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

