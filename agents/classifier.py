"""Email classification agent."""

from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import re

from utils.config import EMAIL_CATEGORIES
from utils.unified_llm_loader import load_llm


class EmailClassification(BaseModel):
    """Email classification output schema."""

    category: str = Field(
        description="The category of the email. Must be one of: technical_support, product_inquiry, billing, feature_request, feedback, unrelated"
    )
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of why this category was chosen")
    priority: str = Field(description="Priority level: high, medium, or low")


class EmailClassifierAgent:
    """Agent responsible for classifying incoming support emails."""

    def __init__(self):
        """Initialize the classifier agent."""
        self.llm = load_llm(temperature=0.3, max_tokens=128)

        self.parser = PydanticOutputParser(pydantic_object=EmailClassification)

        self.prompt_template = """You are an expert email classification agent for TaskFlow Pro, a project management SaaS platform.

Classify the email into ONE of these categories:
{categories}

CLASSIFICATION RULES:
1. technical_support: Login issues, bugs, errors, password resets, sync problems, data issues (will use documentation to help)
2. product_inquiry: Questions about features, pricing, plans, how things work (will answer from documentation)
3. billing: Payment issues, subscription problems, refunds, invoices (ALWAYS escalated to human - no database access)
4. feature_request: Suggestions for new features or improvements (saved for review, no response needed)
5. feedback: Praise, complaints, general comments about the product (saved for review, no response needed)
6. unrelated: Spam, marketing, completely off-topic (skipped entirely)

PRIORITY RULES:
- HIGH: Can't access account, payment failed, critical bug, data loss, billing issues
- MEDIUM: Non-critical questions, minor bugs, integration issues, product questions
- LOW: Feature requests, general feedback, suggestions

EXAMPLES:

Email: "I can't login to my account"
{{
  "category": "technical_support",
  "priority": "high",
  "confidence": 0.95,
  "reasoning": "User cannot access their account - this is a critical technical issue"
}}

Email: "How much does the team plan cost?"
{{
  "category": "product_inquiry",
  "priority": "medium",
  "confidence": 0.90,
  "reasoning": "Customer asking about pricing information"
}}

Email: "I was charged twice this month"
{{
  "category": "billing",
  "priority": "high",
  "confidence": 0.95,
  "reasoning": "Duplicate charge is a billing issue requiring immediate attention"
}}

NOW CLASSIFY THIS EMAIL:

From: {sender}
Subject: {subject}
Body: {body}

Return ONLY a JSON object with these exact fields:
{{
  "category": "one of the categories above",
  "priority": "high, medium, or low",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}

DO NOT return the JSON schema. Return actual values."""

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
        categories_text = "\n".join(
            [f"- {key}: {description}" for key, description in EMAIL_CATEGORIES.items()]
        )

        prompt = self.prompt_template.format(
            categories=categories_text,
            sender=email_data.get("sender", "Unknown"),
            subject=email_data.get("subject", "No subject"),
            body=email_data.get("body", "")[:500],  # Limit body length for better focus
        )

        raw_response = self.llm.invoke(prompt)

        if hasattr(raw_response, "content"):
            response = raw_response.content
        else:
            response = raw_response

        try:
            json_match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result_dict = json.loads(json_str)

                # Check if this is actual data or schema
                if "title" in result_dict and "description" in result_dict:
                    # LLM returned schema instead of data
                    print("Warning: LLM returned JSON schema instead of classification")
                    print(f"Full response: {response[:300]}...")
                    # Fall back to keyword-based classification
                    return self._fallback_classification(email_data)

                return EmailClassification(**result_dict)
            else:
                # Try to parse the entire response
                result_dict = json.loads(response)

                # Check for schema
                if "title" in result_dict:
                    return self._fallback_classification(email_data)

                return EmailClassification(**result_dict)

        except json.JSONDecodeError as e:
            print(f"Error parsing classification JSON: {e}")
            print(f"Response preview: {response[:200]}...")
            return self._fallback_classification(email_data)

        except Exception as e:
            print(f"Error creating classification: {e}")
            print(f"Response type: {type(response)}")
            return self._fallback_classification(email_data)

    def _fallback_classification(self, email_data: Dict[str, Any]) -> EmailClassification:
        """
        Fallback classification using keyword matching when LLM fails.

        Args:
            email_data: Email data dictionary

        Returns:
            EmailClassification based on keywords
        """
        subject = email_data.get("subject", "").lower()
        body = email_data.get("body", "").lower()
        text = subject + " " + body

        # Technical support keywords
        if any(
            word in text
            for word in [
                "login",
                "password",
                "access",
                "error",
                "bug",
                "crash",
                "not working",
                "broken",
                "reset",
                "can't",
                "cannot",
            ]
        ):
            return EmailClassification(
                category="technical_support",
                priority=(
                    "high"
                    if any(word in text for word in ["urgent", "asap", "critical", "can't access"])
                    else "medium"
                ),
                confidence=0.7,
                reasoning="Keyword-based classification: Technical issue detected",
            )

        # Billing keywords
        if any(
            word in text
            for word in [
                "billing",
                "charge",
                "payment",
                "invoice",
                "refund",
                "subscription",
                "cancel",
            ]
        ):
            return EmailClassification(
                category="billing",
                priority=(
                    "high"
                    if any(
                        word in text for word in ["charged twice", "wrong amount", "didn't receive"]
                    )
                    else "medium"
                ),
                confidence=0.75,
                reasoning="Keyword-based classification: Billing issue detected",
            )

        # Feature request keywords
        if any(
            word in text
            for word in [
                "feature request",
                "suggestion",
                "would be nice",
                "please add",
                "could you add",
            ]
        ):
            return EmailClassification(
                category="feature_request",
                priority="low",
                confidence=0.7,
                reasoning="Keyword-based classification: Feature request detected",
            )

        # Feedback keywords
        if any(
            word in text
            for word in ["love", "great", "thank", "feedback", "disappointed", "frustrated"]
        ):
            return EmailClassification(
                category="feedback",
                priority="low",
                confidence=0.65,
                reasoning="Keyword-based classification: Feedback detected",
            )

        # Default to product inquiry
        return EmailClassification(
            category="product_inquiry",
            priority="medium",
            confidence=0.6,
            reasoning="Keyword-based classification: General inquiry (fallback)",
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