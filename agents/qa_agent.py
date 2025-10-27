"""Quality Assurance agent for reviewing email responses."""
from typing import Dict, Any
from langchain_community.llms import HuggingFacePipeline
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import re

from utils.llm_loader import load_llm_pipeline


class QAResult(BaseModel):
    """Quality assurance result schema."""
    approved: bool = Field(
        description="Whether the email response is approved for sending"
    )
    quality_score: float = Field(
        description="Overall quality score from 0 to 10"
    )
    issues: list[str] = Field(
        description="List of issues found (empty if none)"
    )
    suggestions: list[str] = Field(
        description="List of improvement suggestions (empty if none)"
    )
    tone_assessment: str = Field(
        description="Assessment of the email tone (e.g., professional, empathetic, clear)"
    )
    reasoning: str = Field(
        description="Overall reasoning for the approval decision"
    )


class QAAgent:
    """Agent responsible for quality assurance of generated email responses."""
    
    def __init__(self):
        """Initialize the QA agent."""
        pipe = load_llm_pipeline(temperature=0.3)
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.parser = PydanticOutputParser(pydantic_object=QAResult)
        
        # Create QA prompt with examples
        self.prompt = """You are a quality assurance specialist for customer support emails at TaskFlow Pro.

Review the email response and score it on quality (0-10).

QUALITY CRITERIA:
1. Accuracy: Information is correct
2. Completeness: Addresses the customer's question
3. Tone: Professional and empathetic
4. Clarity: Easy to understand
5. Actionability: Provides clear next steps

SCORING:
- 8-10: Excellent, approve immediately
- 7: Good, approve
- 4-6: Needs improvement, reject
- 0-3: Poor, reject

EXAMPLE EVALUATION:

Response: "Thanks for contacting us. We'll look into it."
{{
  "approved": false,
  "quality_score": 3.0,
  "issues": ["Too vague", "No specific help", "Lacks empathy"],
  "suggestions": ["Provide specific steps", "Show understanding"],
  "tone_assessment": "Too brief and impersonal",
  "reasoning": "Response doesn't address the customer's specific issue"
}}

NOW REVIEW THIS EMAIL:

Original Email:
From: {sender}
Subject: {subject}
Category: {category}
Priority: {priority}

{customer_body}

---

Generated Response:
{response}

---

Return ONLY a JSON object with these exact fields:
{{
  "approved": true or false,
  "quality_score": 0.0 to 10.0,
  "issues": ["list", "of", "issues"] or [],
  "suggestions": ["list", "of", "suggestions"] or [],
  "tone_assessment": "description of tone",
  "reasoning": "brief explanation"
}}

DO NOT return the JSON schema. Return actual evaluation values."""
    
    def review(
        self,
        original_email: Dict[str, Any],
        generated_response: str,
        category: str = "general",
        priority: str = "medium"
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
        # Format the prompt (removed format_instructions to avoid schema confusion)
        formatted_prompt = self.prompt.format(
            sender=original_email.get("sender", "Unknown"),
            subject=original_email.get("subject", "No subject"),
            category=category,
            priority=priority,
            customer_body=original_email.get("body", "")[:300],  # Truncate for focus
            response=generated_response[:500]  # Truncate long responses for faster review
        )
        
        # Get LLM response
        llm_response = self.llm.invoke(formatted_prompt)
        
        # Parse the response
        try:
            # Try to extract JSON from the response (improved regex)
            json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result_dict = json.loads(json_str)
                
                # Check if this is actual data or schema (same issue as classifier)
                if 'title' in result_dict and 'description' in result_dict:
                    # LLM returned schema instead of data
                    print("Warning: LLM returned JSON schema instead of QA evaluation")
                    print(f"Full response: {llm_response[:300]}...")
                    # Fall back to simple evaluation
                    return self._fallback_qa_result(generated_response)
                
                result = QAResult(**result_dict)
            else:
                # Try parsing the entire response
                result_dict = json.loads(llm_response)
                
                # Check for schema
                if 'title' in result_dict:
                    return self._fallback_qa_result(generated_response)
                
                result = QAResult(**result_dict)
                
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
        score = 7.0  # Start with passing score
        
        # Length check
        if len(response) < 100:
            issues.append("Response is too short")
            suggestions.append("Provide more detailed assistance")
            score -= 2.0
        
        # Check for professional closing
        if not any(word in response.lower() for word in ["regards", "sincerely", "best"]):
            issues.append("Missing professional closing")
            suggestions.append("Add professional sign-off")
            score -= 1.0
        
        # Check for empathy/greeting
        if not any(word in response.lower() for word in ["thank", "understand", "appreciate", "dear", "hi", "hello"]):
            issues.append("Lacks empathy or greeting")
            suggestions.append("Add empathetic greeting")
            score -= 1.0
        
        # Check for actionable content
        if not any(word in response for word in [":", "1.", "2.", "step", "please", "can", "will"]):
            issues.append("May lack actionable steps")
            suggestions.append("Provide clear action items")
            score -= 0.5
        
        # Ensure score is in valid range
        score = max(0.0, min(10.0, score))
        
        return QAResult(
            approved=score >= 7.0,
            quality_score=score,
            issues=issues if issues else [],
            suggestions=suggestions if suggestions else [],
            tone_assessment="Automated heuristic evaluation (LLM parse failed)",
            reasoning=f"Fallback evaluation based on basic checks. Score: {score:.1f}/10"
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
        
        # Check for basic structure
        if "regards" not in response.lower() and "sincerely" not in response.lower():
            issues.append("Missing professional closing")
        
        # Check for common errors
        if "taskflow pro" in response.lower() and "TaskFlow Pro" not in response:
            issues.append("Product name should be capitalized: 'TaskFlow Pro'")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }


if __name__ == "__main__":
    # Test the QA agent
    print("Testing QA Agent...")
    print("=" * 60)
    
    agent = QAAgent()
    
    # Test case 1: Good response
    test_email_1 = {
        "sender": "john@example.com",
        "subject": "How do I create a project?",
        "body": "I'm new to TaskFlow Pro. How do I create my first project?"
    }
    
    good_response = """Hi John,

Welcome to TaskFlow Pro! I'm happy to help you create your first project.

Here's how to get started:

1. Log in to your TaskFlow Pro account at https://app.taskflowpro.com
2. Click the "New Project" button in the top right corner
3. Enter your project name and description
4. Select a template (Kanban, Scrum, or Waterfall)
5. Click "Create Project"

That's it! You'll be taken to your new project dashboard where you can start adding tasks and inviting team members.

For more detailed guidance, check out our Getting Started guide: https://help.taskflowpro.com/getting-started

Let me know if you have any questions!

Best regards,
TaskFlow Support Team"""
    
    print("\n--- Test 1: Good Response ---")
    print("Running QA review...\n")
    
    try:
        # Quick check first
        quick_result = agent.quick_check(good_response)
        print(f"Quick Check: {'PASSED' if quick_result['passed'] else 'FAILED'}")
        if quick_result['issues']:
            print("Issues:", quick_result['issues'])
        
        # Full review
        result = agent.review(
            original_email=test_email_1,
            generated_response=good_response,
            category="product_inquiry",
            priority="medium"
        )
        
        print(f"\nApproved: {result.approved}")
        print(f"Quality Score: {result.quality_score}/10")
        print(f"Tone: {result.tone_assessment}")
        print(f"Reasoning: {result.reasoning}")
        
        if result.issues:
            print("\nIssues:")
            for issue in result.issues:
                print(f"  - {issue}")
        
        if result.suggestions:
            print("\nSuggestions:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    
    # Test case 2: Poor response
    poor_response = """hey john,

just click the button lol. its easy.

bye"""
    
    print("\n--- Test 2: Poor Response ---")
    print("Running QA review...\n")
    
    try:
        result = agent.review(
            original_email=test_email_1,
            generated_response=poor_response,
            category="product_inquiry",
            priority="medium"
        )
        
        print(f"Approved: {result.approved}")
        print(f"Quality Score: {result.quality_score}/10")
        print(f"Tone: {result.tone_assessment}")
        print(f"Reasoning: {result.reasoning}")
        
        if result.issues:
            print("\nIssues:")
            for issue in result.issues:
                print(f"  - {issue}")
        
        if result.suggestions:
            print("\nSuggestions:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")
        
        if agent.needs_revision(result):
            print("\n" + "-" * 60)
            print("REVISION NEEDED")
            print("-" * 60)
            print(agent.get_revision_prompt(result))
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

