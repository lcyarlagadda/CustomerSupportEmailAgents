"""Quality Assurance agent for reviewing email responses."""
from typing import Dict, Any
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import re

from utils.unified_llm_loader import load_llm


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
        # QA evaluation needs short responses - use 150 tokens
        self.llm = load_llm(temperature=0.5, max_tokens=150)  # Slightly higher for varied evaluations
        
        self.parser = PydanticOutputParser(pydantic_object=QAResult)
        
        # Create QA prompt - simple and direct
        self.prompt = """Evaluate this support email response. Score 0-10 where 8-10=approve, 0-7=reject.

Check: Does it answer the question? Is tone professional? Are steps clear?

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
        # Format the prompt with actual email content
        customer_body = original_email.get("body", "")[:500]  # Show more context now
        response_text = generated_response[:800]  # Show more of the response
        
        formatted_prompt = self.prompt.format(
            customer_body=customer_body,
            response=response_text
        )
        
        # Get LLM response
        llm_response = self.llm.invoke(formatted_prompt)
        
        # Debug: Show first 200 chars of response
        # print(f"[QA Debug] LLM response preview: {llm_response[:200]}...")
        
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
        score = 8.0  # Start with good score
        tone_parts = []
        
        response_lower = response.lower()
        
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
        has_greeting = any(response.startswith(word) or f"\n{word}" in response[:100] for word in ["Dear", "Hi", "Hello", "Hey"])
        if not has_greeting:
            issues.append("Missing greeting")
            suggestions.append("Start with 'Dear [Name]' or 'Hi'")
            score -= 0.8
            tone_parts.append("abrupt start")
        
        # Empathy check - be more specific
        empathy_words = ["understand", "appreciate", "frustrat", "apologize", "sorry", "thank you for"]
        has_empathy = any(word in response_lower for word in empathy_words)
        if not has_empathy and len(response) > 100:
            issues.append("Could show more empathy")
            suggestions.append("Acknowledge customer's concern/frustration")
            score -= 0.7
            tone_parts.append("lacks empathy")
        
        # Professional closing
        closing_words = ["regards", "sincerely", "best", "thank you,", "thanks,"]
        has_closing = any(word in response_lower for word in closing_words)
        if not has_closing:
            issues.append("Missing professional closing")
            suggestions.append("End with 'Best regards' or similar")
            score -= 0.8
            tone_parts.append("incomplete")
        
        # Actionable content - more specific
        has_steps = bool(re.search(r'\d+[.):]', response))  # Numbered steps
        has_bullets = '•' in response or '*' in response
        has_actions = any(word in response_lower for word in ["please", "you can", "try", "click", "go to", "visit"])
        
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
            reasoning=reasoning
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

