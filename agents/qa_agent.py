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
        
        # Create QA prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality assurance specialist for customer support emails at TaskFlow Pro.

Your task is to review generated email responses and ensure they meet our quality standards.

Quality Criteria:
1. **Accuracy**: Information is correct and relevant
2. **Completeness**: Customer's question/issue is fully addressed
3. **Tone**: Professional, empathetic, and friendly
4. **Clarity**: Easy to understand, well-structured
5. **Grammar**: No spelling or grammatical errors
6. **Actionability**: Clear next steps or solutions provided
7. **Appropriateness**: Response matches the customer's issue severity
8. **Brand voice**: Maintains TaskFlow Pro's helpful and professional brand

Approval Guidelines:
- APPROVE (approved=true): Response meets all criteria, quality_score >= 7
- REJECT (approved=false): Significant issues that need fixing, quality_score < 7

When rejecting, provide specific issues and actionable suggestions for improvement.

{format_instructions}"""),
            ("user", """Please review this email response:

Original Customer Email:
From: {sender}
Subject: {subject}
Category: {category}
Priority: {priority}

{customer_body}

---

Generated Response:
{response}

---

Please evaluate this response thoroughly.""")
        ])
    
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
        # Create the chain
        chain = self.prompt | self.llm | self.parser
        
        # Run review
        result = chain.invoke({
            "format_instructions": self.parser.get_format_instructions(),
            "sender": original_email.get("sender", "Unknown"),
            "subject": original_email.get("subject", "No subject"),
            "category": category,
            "priority": priority,
            "customer_body": original_email.get("body", ""),
            "response": generated_response
        })
        
        return result
    
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
        
        # Check length
        if len(response) < 50:
            issues.append("Response is too short")
        
        if len(response) > 5000:
            issues.append("Response is too long")
        
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

