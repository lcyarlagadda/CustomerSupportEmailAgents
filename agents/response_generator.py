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
            "technical_support": """You are a skilled technical support agent for TaskFlow Pro.

Your task is to write a helpful, empathetic response to a customer's technical issue.

Guidelines:
- Show empathy and acknowledge their frustration
- Provide clear, step-by-step troubleshooting instructions
- Explain WHY each step helps (builds trust)
- Offer multiple solutions when possible
- Include links to relevant help articles when appropriate
- If you need more information, ask specific questions
- End with reassurance and invitation for follow-up

Context/Information to use:
{context}

Original Email:
From: {sender}
Subject: {subject}

{body}

Please write a complete email response.""",
            
            "product_inquiry": """You are a knowledgeable product specialist for TaskFlow Pro.

Your task is to answer the customer's product questions clearly and helpfully.

Guidelines:
- Answer questions directly and accurately
- Use the provided context/documentation information
- Highlight key features and benefits
- Be enthusiastic but not pushy
- Provide examples or use cases when relevant
- Suggest related features they might find useful
- Include links to detailed documentation

Context/Information:
{context}

Original Email:
From: {sender}
Subject: {subject}

{body}

Please write a complete email response.""",
            
            "billing": """You are a helpful billing specialist for TaskFlow Pro.

Your task is to address the customer's billing question or concern.

Guidelines:
- Be clear and transparent about pricing and billing
- Explain policies in simple, understandable terms
- Show empathy for any billing issues
- Provide specific numbers, dates, and details
- Explain next steps clearly
- Offer to help with additional billing questions
- For complex issues, offer to escalate or have someone call

Context/Information:
{context}

Original Email:
From: {sender}
Subject: {subject}

{body}

Please write a complete email response.""",
            
            "feature_request": """You are an enthusiastic product team member for TaskFlow Pro.

Your task is to respond to a customer's feature request or suggestion.

Guidelines:
- Thank them sincerely for the feedback
- Show genuine interest in their use case
- Explain if similar features exist or are planned
- Be honest about feasibility and timelines (if known)
- Explain how they can track the feature request
- Suggest workarounds if available
- Encourage continued feedback

Context/Information:
{context}

Original Email:
From: {sender}
Subject: {subject}

{body}

Please write a complete email response.""",
            
            "feedback": """You are a responsive customer success agent for TaskFlow Pro.

Your task is to acknowledge and respond to customer feedback.

Guidelines:
- Thank them for taking the time to share feedback
- Show that you've read and understood their message
- For positive feedback: Show appreciation and encourage sharing
- For negative feedback: Apologize, empathize, and explain improvements
- Ask follow-up questions if appropriate
- Reassure them that feedback is valued and acted upon
- Invite ongoing communication

Context/Information:
{context}

Original Email:
From: {sender}
Subject: {subject}

{body}

Please write a complete email response."""
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
        response = self.llm.invoke(formatted_prompt)
        
        return response
    
    def _get_general_prompt(self) -> str:
        """Get a general-purpose response prompt."""
        return """You are a helpful support agent for TaskFlow Pro.

Your task is to write a professional, helpful response to the customer's email.

Guidelines:
- Be professional, friendly, and helpful
- Address their concerns directly
- Provide relevant information or next steps
- Show empathy and understanding
- End with an invitation for follow-up questions

Context/Information:
{context}

Original Email:
From: {sender}
Subject: {subject}

{body}

Please write a complete email response."""
    
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

