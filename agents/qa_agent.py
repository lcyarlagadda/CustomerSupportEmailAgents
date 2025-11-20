"""Quality Assurance agent for reviewing email responses with structured outputs."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
import json
import re

from utils.instructor_llm import get_structured_response


class QAResult(BaseModel):
    """Quality assurance result schema."""

    approved: bool = Field(description="Whether the email response is approved for sending")
    quality_score: float = Field(description="Overall quality score from 0 to 10", ge=0, le=10)
    issues: List[str] = Field(default_factory=list, description="List of issues found (empty if none)")
    suggestions: List[str] = Field(default_factory=list, description="List of improvement suggestions (empty if none)")
    tone_assessment: str = Field(
        description="Assessment of the email tone (e.g., professional, empathetic, clear)"
    )
    reasoning: str = Field(description="Overall reasoning for the approval decision")


class QAAgent:
    """Agent responsible for quality assurance of generated email responses with structured evaluation."""

    def __init__(self):
        """Initialize the QA agent."""
        self.prompt = """Evaluate this support email response. Score 0-10 where 8-10=approve, 0-7=reject.

EVALUATION RULES:
- If response escalates to specialist/team (billing, technical, etc.), automatically approve with score 9-10
- If response says "need more information" or "escalate", that is ACCEPTABLE, score 9+
- Professional, calm tone is REQUIRED regardless of customer tone
- If response provides clear answer/steps from documentation, score 9-10
- Only reject if: unprofessional, makes up information, or has major errors

CUSTOMER'S EMAIL:
{customer_body}

OUR RESPONSE:
{response}

Output ONLY this JSON (fill with your evaluation):
{{"approved": true/false, "quality_score": X.X, "issues": ["issue1", "issue2"], "suggestions": ["fix1", "fix2"], "tone_assessment": "describe tone", "reasoning": "why approve/reject"}}

Evaluate THIS specific response above. Do not use examples."""

    def review(
        self,
        original_email: Dict[str, Any],
        generated_response: str,
        category: str = "general",
        priority: str = "medium",
    ) -> QAResult:
        """
        Review a generated email response.

        Args:
            original_email: Dictionary with original email data
                - sender: Sender's email
                - subject: Email subject
                - body: Email body
            generated_response: The AI-generated response to review
            category: Email category
            priority: Email priority

        Returns:
            QAResult object with evaluation
        """
        # Format the prompt with actual email content
        customer_body = original_email.get("body", "")[:500]  # Show more context now
        response_text = generated_response[:800]  # Show more of the response

        formatted_prompt = self.prompt.format(customer_body=customer_body, response=response_text)

        try:
            # Use instructor for structured output - automatically validates and returns Pydantic model
            qa_result = get_structured_response(
                prompt=formatted_prompt,
                response_model=QAResult,
                temperature=0.5,
                max_tokens=300
            )
            return qa_result

        except json.JSONDecodeError as e:
            print(f"Error parsing QA JSON: {e}")
            print(f"Response preview: {llm_response[:300]}...")
            return self._fallback_qa_result(generated_response)

        except Exception as e:
            print(f"Error creating QA result: {e}")
            print(f"Response type: {type(llm_response)}")
            return self._fallback_qa_result(generated_response)

        return result

    def _fallback_qa_result(self, response: str) -> QAResult:
        """
        Fallback QA evaluation using simple heuristics when LLM fails.

        Args:
            response: Generated email response

        Returns:
            QAResult based on basic checks
        """
        issues = []
        suggestions = []
        score = 8.0
        tone_parts = []

        response_lower = response.lower()

        escalation_keywords = [
            "escalat",
            "specialist",
            "billing team",
            "technical team",
            "product team",
            "review by",
            "forward to",
            "manual review",
            "need to connect you",
            "reach out to you directly",
        ]
        is_escalation = any(keyword in response_lower for keyword in escalation_keywords)

        if is_escalation:
            return QAResult(
                approved=True,
                quality_score=9.0,
                issues=[],
                suggestions=[],
                tone_assessment="Professional escalation response",
                reasoning="Escalation to appropriate team is acceptable and professional",
            )

        # Length checks - more nuanced
        if len(response) < 80:
            issues.append("Response too brief - needs more detail")
            suggestions.append("Expand with specific steps or information")
            score -= 3.0
            tone_parts.append("too brief")
        elif len(response) < 150:
            issues.append("Response somewhat short")
            suggestions.append("Add more context or details")
            score -= 1.0
        elif len(response) > 3000:
            issues.append("Response very long")
            suggestions.append("Condense to key points")
            score -= 1.5
            tone_parts.append("overly verbose")

        # Greeting check
        has_greeting = any(
            response.startswith(word) or f"\n{word}" in response[:100]
            for word in ["Dear", "Hi", "Hello", "Hey"]
        )
        if not has_greeting:
            issues.append("Missing greeting")
            suggestions.append("Start with 'Dear [Name]' or 'Hi'")
            score -= 0.8
            tone_parts.append("abrupt start")

        # Empathy check - be more specific
        empathy_words = [
            "understand",
            "appreciate",
            "frustrat",
            "apologize",
            "sorry",
            "thank you for",
        ]
        has_empathy = any(word in response_lower for word in empathy_words)
        if not has_empathy and len(response) > 100:
            issues.append("Could show more empathy")
            suggestions.append("Acknowledge customer's concern/frustration")
            score -= 0.7
            tone_parts.append("lacks empathy")

        # Check for required closing
        required_closing = "hope you have a great day"
        has_proper_closing = required_closing in response_lower
        if not has_proper_closing:
            issues.append("Missing required closing: 'Hope you have a great day!'")
            suggestions.append("End with 'Hope you have a great day!' before signature")
            score -= 1.0
            tone_parts.append("missing proper closing")

        # Actionable content - more specific
        has_steps = bool(re.search(r"\d+[.):]", response))  # Numbered steps
        has_bullets = "•" in response or "*" in response
        has_actions = any(
            word in response_lower
            for word in ["please", "you can", "try", "click", "go to", "visit"]
        )

        if not (has_steps or has_bullets or has_actions):
            issues.append("Lacks clear action items")
            suggestions.append("Provide specific steps or next actions")
            score -= 1.0
            tone_parts.append("vague")

        # Check for common issues
        if response.count("regards") > 2:
            issues.append("Multiple sign-offs detected")
            score -= 0.5

        if "guidelines:" in response_lower or "context:" in response_lower:
            issues.append("Contains prompt artifacts")
            suggestions.append("Remove system instructions from response")
            score -= 2.0
            tone_parts.append("contains instructions")

        # Build tone assessment
        if not tone_parts:
            if score >= 8.0:
                tone_assessment = "Professional, empathetic, and helpful"
            elif score >= 7.0:
                tone_assessment = "Acceptable professional tone"
            else:
                tone_assessment = "Professional but could be improved"
        else:
            tone_assessment = f"Issues detected: {', '.join(tone_parts)}"

        # Build reasoning
        if not issues:
            reasoning = "Response meets basic quality standards"
        else:
            reasoning = f"Found {len(issues)} issue(s): {'; '.join(issues[:2])}"

        # Ensure score is in valid range
        score = max(0.0, min(10.0, score))

        return QAResult(
            approved=score >= 7.0,
            quality_score=round(score, 1),
            issues=issues if issues else [],
            suggestions=suggestions if suggestions else [],
            tone_assessment=tone_assessment,
            reasoning=reasoning,
        )

    def needs_revision(self, qa_result: QAResult) -> bool:
        """
        Determine if a response needs revision.

        Args:
            qa_result: QAResult from review

        Returns:
            True if revision needed, False otherwise
        """
        return not qa_result.approved or qa_result.quality_score < 7.0

    def get_revision_prompt(self, qa_result: QAResult) -> str:
        """
        Generate a prompt for revising a rejected response.

        Args:
            qa_result: QAResult with issues and suggestions

        Returns:
            Revision prompt string
        """
        prompt = "Please revise the response to address the following issues:\n\n"

        if qa_result.issues:
            prompt += "Issues to fix:\n"
            for i, issue in enumerate(qa_result.issues, 1):
                prompt += f"{i}. {issue}\n"

        if qa_result.suggestions:
            prompt += "\nSuggestions for improvement:\n"
            for i, suggestion in enumerate(qa_result.suggestions, 1):
                prompt += f"{i}. {suggestion}\n"

        return prompt

    def quick_check(self, response: str) -> Dict[str, Any]:
        """
        Perform a quick sanity check without full LLM evaluation.

        Args:
            response: Email response to check

        Returns:
            Dictionary with quick check results
        """
        issues = []

        # Check length (reasonable limits for customer support emails)
        if len(response) < 50:
            issues.append("Response is too short")

        # Note: Don't penalize longer responses if they're comprehensive
        # Only flag if extremely long
        if len(response) > 3000:
            issues.append("Response is very long - consider breaking into multiple emails")

        # Check for placeholder text
        placeholders = ["[PLACEHOLDER]", "[TODO]", "[INSERT", "XXX", "TBD"]
        for placeholder in placeholders:
            if placeholder.lower() in response.lower():
                issues.append(f"Contains placeholder text: {placeholder}")

        # Check for required closing
        if "hope you have a great day" not in response.lower():
            issues.append("Missing required closing: 'Hope you have a great day!'")

        # Check for common errors
        if "taskflow pro" in response.lower() and "TaskFlow Pro" not in response:
            issues.append("Product name should be capitalized: 'TaskFlow Pro'")

        return {"passed": len(issues) == 0, "issues": issues}