"""Response generation agent for crafting email replies."""

from typing import Dict, Any, Optional

from utils.unified_llm_loader import load_llm


class ResponseGeneratorAgent:
    """Agent responsible for generating email responses."""

    def __init__(self):
        """Initialize the response generator agent."""
        self.llm = load_llm(temperature=0.8, max_tokens=400)

        self.response_format = """
        REQUIRED FORMAT (follow exactly):

        1. Salutation: "Hi [Name]," or "Hello [Name],"

        2. Acknowledgment:
        - If issue/problem: Acknowledge and apologize
        - If feedback/suggestion/request: Appreciate their input

        3. Main content: [Your specific response]

        4. Closing: "Hope you have a great day!"

        5. DO NOT include signature - it will be added automatically

        Example structure:
        ---
        Hi [Name],

        [Acknowledgment sentence]

        [Main content - solution, information, or action taken]

        Hope you have a great day!
        ---
        """

        self.prompts = {
            "technical_support": """You are a professional support agent for TaskFlow Pro. A customer needs technical help.

Customer Email:
Subject: {subject}
Body: {body}

Documentation Available: {context}

{format_instructions}

SPECIFIC INSTRUCTIONS:
- Acknowledge their technical issue and apologize for the inconvenience
- If documentation contains solution: Provide clear step-by-step instructions
- If documentation does NOT contain solution: Apologize and explain you are escalating to a technical specialist who will contact them within 24 hours
- Never make up information or promise things not in the documentation
- Stay calm and professional regardless of customer tone
- Keep response 200-300 words

Write ONLY the email body (no signature):""",
            "product_inquiry": """You are a professional support agent for TaskFlow Pro. A customer has a product question.

Customer Email:
Subject: {subject}
Body: {body}

Documentation Available: {context}

{format_instructions}

SPECIFIC INSTRUCTIONS:
- Thank them for their question
- If documentation has the answer: Provide clear, accurate information
- If NOT in documentation: Apologize and explain you are escalating to the product team for accurate information, they will respond within 24 hours
- Never make up features, pricing, or capabilities
- Stay professional and informative
- Keep response 200-300 words

Write ONLY the email body (no signature):""",
            "billing": """You are a professional support agent for TaskFlow Pro. A customer has a billing concern.

Customer Email:
Subject: {subject}
Body: {body}

{format_instructions}

SPECIFIC INSTRUCTIONS:
- Acknowledge their billing concern and sincerely apologize for any inconvenience
- Explain that billing matters require direct database access and specialist review
- Assure them you are escalating to the billing team who will contact them within 24 hours with a resolution
- Never attempt to look up account details, process refunds, or make billing promises
- Stay empathetic and reassuring
- Keep response 150-250 words

Write ONLY the email body (no signature):""",
            "feature_request": """You are a professional support agent for TaskFlow Pro. A customer has suggested a feature.

Customer Email:
Subject: {subject}
Body: {body}

{format_instructions}

SPECIFIC INSTRUCTIONS:
- Thank them sincerely for their feature suggestion
- Acknowledge the value of their input
- Explain their request has been saved and will be reviewed by the product team
- Do NOT promise implementation or timelines
- Stay professional and appreciative
- Keep response brief: 100-150 words

Write ONLY the email body (no signature):""",
            "feedback": """You are a professional support agent for TaskFlow Pro. A customer provided feedback.

Customer Email:
Subject: {subject}
Body: {body}

{format_instructions}

SPECIFIC INSTRUCTIONS:
- If positive feedback: Thank them sincerely for their kind words
- If negative feedback: Acknowledge their concerns and apologize professionally, never defensive
- Explain their feedback has been saved and will be reviewed by leadership
- Stay calm and professional even if customer is frustrated
- Keep response brief: 100-150 words

Write ONLY the email body (no signature):""",
        }

    def generate_response(
        self, email_data: Dict[str, Any], category: str, context: Optional[str] = None
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
            context_text = (
                "No additional context provided. Use your general knowledge of TaskFlow Pro."
            )

        # Format prompt with data
        if "{format_instructions}" in prompt:
            formatted_prompt = prompt.format(
                sender=email_data.get("sender", "Valued Customer"),
                subject=email_data.get("subject", "Your inquiry"),
                body=email_data.get("body", ""),
                context=context_text,
                format_instructions=self.response_format,
            )
        else:
            formatted_prompt = prompt.format(
                sender=email_data.get("sender", "Valued Customer"),
                subject=email_data.get("subject", "Your inquiry"),
                body=email_data.get("body", ""),
                context=context_text,
            )

        # Generate response
        raw_response = self.llm.invoke(formatted_prompt)

        if hasattr(raw_response, "content"):
            response_text = raw_response.content
        else:
            response_text = raw_response

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
            "Thanks,",
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

        response = re.sub(r"\[Your [^\]]+\]", "", response)
        response = re.sub(r"\[Your [^\]]+\]\s*\(optional\)", "", response)
        response = re.sub(r"\(optional\)", "", response)

        # Remove meta-commentary and notes (but keep "Hope you have a great day!")
        meta_patterns = [
            r"\n\nNote:.*$",
            r"\n\nPlease note.*$",
            r"\n\nPlease let me know.*$",
            r"\n\nThis response.*$",
            r"\n\nThe response.*$",
            r"\n\nFeel free to.*$",
            r"\n\nIf you need any modifications.*$",
            r"\n\n---.*$",
        ]

        for pattern in meta_patterns:
            response = re.sub(pattern, "", response, flags=re.DOTALL | re.IGNORECASE)

        # Remove any remaining prompt artifacts
        prompt_artifacts = [
            "You are a skilled technical support agent",
            "Your task is to write",
            "Guidelines:",
            "Context/Information to use:",
            "Original Email:",
            "Write a complete but CONCISE",
            "Use the relevant information from",
            "Customer Email -",
        ]

        for artifact in prompt_artifacts:
            if artifact in response[:200]:  # Only check beginning
                # This is likely still the prompt, try harder to find actual response
                lines = response.split("\n")
                for i, line in enumerate(lines):
                    if any(marker in line for marker in start_markers):
                        response = "\n".join(lines[i:])
                        break

        # Clean up extra whitespace, tabs, and normalize indentation
        lines = response.split("\n")
        cleaned_lines = []
        for line in lines:
            # Remove ALL leading whitespace (spaces, tabs, non-breaking spaces)
            # This ensures text is completely left-aligned
            cleaned_line = line.lstrip(" \t\u00a0").rstrip(" \t")
            if cleaned_line:  # Keep non-empty lines
                cleaned_lines.append(cleaned_line)
            elif cleaned_lines:  # Preserve single blank lines between paragraphs
                if cleaned_lines[-1] != "":
                    cleaned_lines.append("")

        # Join lines and normalize paragraph spacing (single blank line between paragraphs)
        response = "\n".join(cleaned_lines)
        # Normalize multiple blank lines to single blank line
        response = re.sub(r"\n{3,}", "\n\n", response)

        # Final strip to ensure no leading/trailing whitespace on entire response
        return response.strip()

    def _get_general_prompt(self) -> str:
        """Get a general-purpose response prompt."""
        return """Customer Email: {subject}
{body}

Available Info: {context}

Write ONLY the email body (200-400 words). Start with greeting, end BEFORE signature.
DO NOT include: [placeholders], notes, explanations, signature, or "please let me know"."""

    def add_signature(self, email_body: str) -> str:
        """
        Add a professional signature to the email.

        Args:
            email_body: The email body content (should NOT already have a signature)

        Returns:
            Email with signature
        """
        signature_check = ["Thanks,", "TaskFlow Pro Team", "support@taskflowpro.com"]
        if any(sig in email_body for sig in signature_check):
            # Already has signature, but clean it up - remove ALL leading whitespace
            lines = email_body.split("\n")
            cleaned_lines = []
            for line in lines:
                cleaned_line = line.lstrip(" \t\u00a0").rstrip(" \t")
                cleaned_lines.append(cleaned_line)
            return "\n".join(cleaned_lines)

        # Remove all leading whitespace (spaces, tabs, non-breaking spaces) from each line
        # This ensures perfect left-alignment
        lines = email_body.split("\n")
        cleaned_lines = []
        for line in lines:
            # Remove ALL leading whitespace to ensure left alignment
            cleaned_line = line.lstrip(" \t\u00a0").rstrip(" \t")
            cleaned_lines.append(cleaned_line)
        email_body = "\n".join(cleaned_lines)
        email_body = email_body.rstrip()

        # Ensure proper spacing before signature
        if not email_body.endswith("\n\n"):
            if email_body.endswith("\n"):
                email_body += "\n"
            else:
                email_body += "\n\n"

        # Add signature with no tabs or extra whitespace
        signature = "Thanks,\nTaskFlow Pro Team\nsupport@taskflowpro.com"

        return email_body + signature
