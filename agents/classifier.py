"""Email classification agent."""
from typing import Dict, Any
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import re

from utils.config import EMAIL_CATEGORIES
from utils.llm_loader import load_llm_pipeline


class EmailClassification(BaseModel):
    """Email classification output schema."""
    category: str = Field(
        description="The category of the email. Must be one of: technical_support, product_inquiry, billing, feature_request, feedback, unrelated"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen"
    )
    priority: str = Field(
        description="Priority level: high, medium, or low"
    )


class EmailClassifierAgent:
    """Agent responsible for classifying incoming support emails."""
    
    def __init__(self):
        """Initialize the classifier agent."""
        pipe = load_llm_pipeline(temperature=0.7)
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.parser = PydanticOutputParser(pydantic_object=EmailClassification)
        
        # Create prompt template
        self.prompt_template = """You are an expert email classification agent for TaskFlow Pro, a project management SaaS platform.

Your task is to analyze incoming support emails and classify them into the appropriate category.

Available categories:
{categories}

Important guidelines:
- Choose the MOST SPECIFIC category that matches the email's primary intent
- If an email contains multiple topics, classify based on the MAIN issue
- Mark emails as "unrelated" if they are spam, marketing, or completely off-topic
- Set priority based on urgency and business impact:
  * HIGH: Account access issues, billing problems, critical bugs affecting work
  * MEDIUM: Feature questions, minor bugs, integration issues
  * LOW: Feature requests, general feedback, how-to questions

{format_instructions}

Please classify this email:

From: {sender}
Subject: {subject}

Body:
{body}

Respond ONLY with valid JSON matching the schema above."""
    
    def classify(self, email_data: Dict[str, Any]) -> EmailClassification:
        """
        Classify an email into a support category.
        
        Args:
            email_data: Dictionary containing email information
                - sender: Email sender address
                - subject: Email subject
                - body: Email body content
        
        Returns:
            EmailClassification object
        """
        categories_text = "\n".join([
            f"- {key}: {description}"
            for key, description in EMAIL_CATEGORIES.items()
        ])
        
        prompt = self.prompt_template.format(
            categories=categories_text,
            format_instructions=self.parser.get_format_instructions(),
            sender=email_data.get("sender", "Unknown"),
            subject=email_data.get("subject", "No subject"),
            body=email_data.get("body", "")
        )
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response (improved regex for nested objects)
            json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result_dict = json.loads(json_str)
                return EmailClassification(**result_dict)
            else:
                # Try to parse the entire response
                result_dict = json.loads(response)
                return EmailClassification(**result_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing classification JSON: {e}")
            print(f"Response preview: {response[:200]}...")
            # Return default classification
            return EmailClassification(
                category="product_inquiry",
                confidence=0.5,
                reasoning="Failed to parse LLM response",
                priority="medium"
            )
        except Exception as e:
            print(f"Error creating classification: {e}")
            return EmailClassification(
                category="product_inquiry",
                confidence=0.5,
                reasoning="Failed to parse LLM response",
                priority="medium"
            )
    
    def should_process(self, classification: EmailClassification) -> bool:
        """
        Determine if an email should be processed further.
        
        Args:
            classification: EmailClassification result
        
        Returns:
            True if email should be processed, False if it should be ignored
        """
        # Don't process unrelated emails
        if classification.category == "unrelated":
            return False
        
        # Don't process low confidence classifications
        if classification.confidence < 0.5:
            return False
        
        return True


if __name__ == "__main__":
    # Test the classifier
    print("Testing Email Classifier Agent...")
    print("=" * 60)
    
    # Test emails
    test_emails = [
        {
            "sender": "john@example.com",
            "subject": "Can't reset password",
            "body": "I'm not receiving password reset emails. Please help urgently!"
        },
        {
            "sender": "sarah@company.com",
            "subject": "How does Slack integration work?",
            "body": "I'd like to know more about integrating TaskFlow Pro with Slack. What features are available?"
        },
        {
            "sender": "spam@marketing.com",
            "subject": "Buy our amazing product!!!",
            "body": "Limited time offer! Click here now!!!"
        }
    ]
    
    agent = EmailClassifierAgent()
    
    for i, email in enumerate(test_emails, 1):
        print(f"\n--- Test Email {i} ---")
        print(f"From: {email['sender']}")
        print(f"Subject: {email['subject']}")
        
        try:
            result = agent.classify(email)
            print(f"\nClassification:")
            print(f"  Category: {result.category}")
            print(f"  Priority: {result.priority}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Reasoning: {result.reasoning}")
            print(f"  Should process: {agent.should_process(result)}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 60)

