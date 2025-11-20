"""
Quality Assurance Agent with Integrated Safety Guardrails

This agent performs comprehensive quality and safety validation:
1. Quality checks (tone, structure, completeness)
2. PII detection and redaction (using Presidio)
3. Toxic language detection (LLM-based)
4. Topic relevance validation
5. Content safety checks
"""

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
import json
import re

from utils.instructor_llm import get_structured_response


@dataclass
class SafetyViolation:
    """Represents a safety/guardrail violation."""
    check_type: str  # "pii", "toxic", "off_topic", "moderation"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    field: Optional[str] = None
    action: str = "flag"  # "flag", "block", "redact"


class QAResult(BaseModel):
    """Quality assurance result schema with integrated safety checks."""

    # Quality assessment
    approved: bool = Field(description="Whether the email response is approved for sending")
    quality_score: float = Field(description="Overall quality score from 0 to 10", ge=0, le=10)
    issues: List[str] = Field(default_factory=list, description="List of quality issues found")
    suggestions: List[str] = Field(default_factory=list, description="List of improvement suggestions")
    tone_assessment: str = Field(description="Assessment of the email tone")
    reasoning: str = Field(description="Overall reasoning for the approval decision")
    
    # Safety assessment (added from guardrails)
    safety_violations: List[str] = Field(default_factory=list, description="List of safety violations")
    should_block: bool = Field(default=False, description="Whether response should be blocked")
    requires_redaction: bool = Field(default=False, description="Whether PII needs redaction")


class QAAgent:
    """
    Comprehensive Quality Assurance Agent with integrated safety guardrails.
    
    Combines quality checks and safety validations:
    - Response quality (tone, structure, completeness)
    - PII detection and redaction (Presidio)
    - Toxic language detection (LLM-based)
    - Topic relevance
    - Content safety
    """

    def __init__(
        self,
        enable_pii_detection: bool = True,
        enable_toxic_detection: bool = True,
        enable_topic_validation: bool = True
    ):
        """
        Initialize the QA agent with safety features.
        
        Args:
            enable_pii_detection: Enable PII detection
            enable_toxic_detection: Enable toxic language detection
            enable_topic_validation: Enable topic relevance validation
        """
        self.enable_pii_detection = enable_pii_detection
        self.enable_toxic_detection = enable_toxic_detection
        self.enable_topic_validation = enable_topic_validation
        
        # Initialize Presidio for PII detection
        self.presidio_available = False
        self.analyzer = None
        self.anonymizer = None
        
        if self.enable_pii_detection:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                self.presidio_available = True
                print("✓ QA Agent: Presidio PII detection initialized")
            except ImportError:
                print("⚠ QA Agent: Presidio not available, using regex fallback")
            except Exception as e:
                print(f"⚠ QA Agent: Presidio initialization failed: {e}")
        
        # Initialize LLM for safety checks
        self.llm = None
        try:
            from utils.unified_llm_loader import load_llm
            self.llm = load_llm(temperature=0.0, model_name="llama-3.1-8b-instant")
            print("✓ QA Agent: LLM-based safety checks initialized")
        except Exception as e:
            print(f"⚠ QA Agent: LLM initialization failed: {e}")
        
        # Fallback PII regex patterns
        self.pii_patterns = {
            "EMAIL_ADDRESS": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "PHONE_NUMBER": re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'),
            "CREDIT_CARD": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            "US_SSN": re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
        }
        
        # Quality evaluation prompt
        self.quality_prompt = """Evaluate this support email response. Score 0-10 where 8-10=approve, 0-7=reject.

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
    ) -> Tuple[QAResult, Optional[str]]:
        """
        Comprehensive review with quality and safety checks.
        
        Args:
            original_email: Dictionary with original email data
            generated_response: The AI-generated response to review
            category: Email category
            priority: Email priority
            
        Returns:
            Tuple of (QAResult, redacted_response or None)
        """
        safety_violations = []
        redacted_response = generated_response
        
        # Step 1: Safety checks (critical - must pass before quality review)
        
        # 1a. PII Detection and Redaction
        if self.enable_pii_detection:
            pii_violations, redacted = self._check_and_redact_pii(generated_response)
            if pii_violations:
                safety_violations.extend([v.message for v in pii_violations])
                redacted_response = redacted
        
        # 1b. Toxic Language Check
        if self.enable_toxic_detection:
            toxic_violations = self._check_toxic_language(generated_response)
            if toxic_violations:
                safety_violations.extend([v.message for v in toxic_violations])
        
        # 1c. Topic Relevance Check
        if self.enable_topic_validation and category:
            topic_violations = self._check_topic_relevance(
                generated_response, 
                original_email.get("body", ""),
                category
            )
            if topic_violations:
                safety_violations.extend([v.message for v in topic_violations])
        
        # 1d. Additional Safety Checks
        additional_violations = self._additional_safety_checks(generated_response)
        if additional_violations:
            safety_violations.extend([v.message for v in additional_violations])
        
        # Step 2: Quality evaluation
        customer_body = original_email.get("body", "")[:500]
        response_text = generated_response[:800]
        
        formatted_prompt = self.quality_prompt.format(
            customer_body=customer_body, 
            response=response_text
        )
        
        try:
            # Get structured quality assessment
            quality_result = get_structured_response(
                prompt=formatted_prompt,
                response_model=QAResult,
                temperature=0.5,
                max_tokens=300
            )
            
            # Add safety information to result
            quality_result.safety_violations = safety_violations
            quality_result.requires_redaction = (redacted_response != generated_response)
            
            # Determine if response should be blocked
            critical_violations = [
                v for v in (
                    (pii_violations if self.enable_pii_detection else []) +
                    (toxic_violations if self.enable_toxic_detection else []) +
                    additional_violations
                )
                if v.severity == "critical" or v.action == "block"
            ]
            
            quality_result.should_block = len(critical_violations) > 0
            
            # If safety violations exist, reduce approval
            if safety_violations and not quality_result.should_block:
                quality_result.issues.extend(safety_violations)
                quality_result.quality_score = min(quality_result.quality_score, 7.0)
            
            # Block if critical safety issues
            if quality_result.should_block:
                quality_result.approved = False
                quality_result.quality_score = 0.0
                quality_result.reasoning = "Response blocked due to critical safety violations"
            
            return quality_result, redacted_response if quality_result.requires_redaction else None
            
        except Exception as e:
            print(f"Error in QA evaluation: {e}")
            fallback_result = self._fallback_qa_result(generated_response, safety_violations)
            return fallback_result, redacted_response if redacted_response != generated_response else None

    def _check_and_redact_pii(self, text: str) -> Tuple[List[SafetyViolation], str]:
        """Check for PII and return violations and redacted text."""
        violations = []
        redacted_text = text
        
        if self.presidio_available and self.analyzer and self.anonymizer:
            try:
                # Analyze text for PII
                results = self.analyzer.analyze(
                    text=text,
                    language='en',
                    entities=[
                        "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", 
                        "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE",
                        "IBAN_CODE", "IP_ADDRESS"
                    ]
                )
                
                if results:
                    # Create violations
                    for result in results:
                        severity = "critical" if result.entity_type in ["CREDIT_CARD", "US_SSN", "US_PASSPORT"] else "high"
                        violations.append(SafetyViolation(
                            check_type=f"pii_{result.entity_type.lower()}",
                            severity=severity,
                            message=f"PII detected: {result.entity_type}",
                            field="response",
                            action="redact"
                        ))
                    
                    # Anonymize the text
                    anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
                    redacted_text = anonymized_result.text
                
                return violations, redacted_text
                
            except Exception as e:
                print(f"⚠ Presidio error: {e}, falling back to regex")
        
        # Fallback: Regex-based detection and redaction
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                severity = "critical" if pii_type in ["CREDIT_CARD", "US_SSN"] else "high"
                violations.append(SafetyViolation(
                    check_type=f"pii_{pii_type.lower()}",
                    severity=severity,
                    message=f"PII detected: {pii_type}",
                    field="response",
                    action="redact"
                ))
                
                # Redact
                if pii_type == "EMAIL_ADDRESS":
                    redacted_text = pattern.sub("[EMAIL REDACTED]", redacted_text)
                elif pii_type == "PHONE_NUMBER":
                    redacted_text = pattern.sub("[PHONE REDACTED]", redacted_text)
                elif pii_type == "CREDIT_CARD":
                    redacted_text = pattern.sub("[CARD REDACTED]", redacted_text)
                elif pii_type == "US_SSN":
                    redacted_text = pattern.sub("[SSN REDACTED]", redacted_text)
        
        return violations, redacted_text

    def _check_toxic_language(self, text: str) -> List[SafetyViolation]:
        """Check for toxic/abusive language using LLM."""
        violations = []
        
        if self.llm:
            try:
                prompt = f"""Analyze the following customer support response for toxic, abusive, or inappropriate language.

Text: "{text[:500]}"

Consider:
- Profanity or explicit language
- Personal attacks or insults
- Threatening language
- Discriminatory content
- Aggressive or hostile tone

Respond with ONLY one of: SAFE, MILD, MODERATE, SEVERE

Response:"""
                
                response = self.llm.invoke(prompt)
                result = response.content.strip().upper()
                
                if result == "SEVERE":
                    violations.append(SafetyViolation(
                        check_type="toxic_language",
                        severity="critical",
                        message="Severe toxic language detected",
                        action="block"
                    ))
                elif result == "MODERATE":
                    violations.append(SafetyViolation(
                        check_type="toxic_language",
                        severity="high",
                        message="Moderate toxic language detected",
                        action="block"
                    ))
                elif result == "MILD":
                    violations.append(SafetyViolation(
                        check_type="unprofessional_language",
                        severity="medium",
                        message="Mild unprofessional language detected",
                        action="flag"
                    ))
                
                return violations
                
            except Exception as e:
                print(f"⚠ LLM toxic check failed: {e}, using fallback")
        
        # Fallback: Pattern-based checks
        return self._check_toxic_patterns(text)

    def _check_toxic_patterns(self, text: str) -> List[SafetyViolation]:
        """Fallback pattern-based toxic language detection."""
        violations = []
        text_lower = text.lower()
        
        offensive_keywords = [
            "stupid", "idiot", "moron", "dumb", "hate", "worst",
            "useless", "garbage", "trash", "pathetic", "incompetent"
        ]
        
        found_keywords = [kw for kw in offensive_keywords if kw in text_lower]
        
        if len(found_keywords) >= 2:
            violations.append(SafetyViolation(
                check_type="toxic_language",
                severity="high",
                message=f"Unprofessional terms detected: {', '.join(found_keywords[:2])}",
                action="flag"
            ))
        elif found_keywords:
            violations.append(SafetyViolation(
                check_type="unprofessional_language",
                severity="medium",
                message=f"Unprofessional language: {found_keywords[0]}",
                action="flag"
            ))
        
        return violations

    def _check_topic_relevance(
        self, 
        response: str, 
        email_context: str, 
        category: str
    ) -> List[SafetyViolation]:
        """Check if response is relevant to the email using LLM."""
        violations = []
        
        if self.llm and len(email_context) > 20:
            try:
                prompt = f"""Determine if the support response is relevant to the customer's email.

Customer Email: "{email_context[:300]}"
Category: {category}
Support Response: "{response[:300]}"

Is the response relevant and addressing the customer's concern?

Respond with ONLY one of: RELEVANT, SOMEWHAT_RELEVANT, OFF_TOPIC

Response:"""
                
                result = self.llm.invoke(prompt)
                relevance = result.content.strip().upper()
                
                if relevance == "OFF_TOPIC":
                    violations.append(SafetyViolation(
                        check_type="off_topic",
                        severity="high",
                        message=f"Response appears off-topic for category: {category}",
                        action="flag"
                    ))
                
                return violations
                
            except Exception:
                pass
        
        return violations

    def _additional_safety_checks(self, response: str) -> List[SafetyViolation]:
        """Additional safety checks."""
        violations = []
        
        # Check for placeholder text
        placeholders = [
            "[INSERT", "[TODO", "{{", "}}", "[NAME]", "[COMPANY]", 
            "XXX", "[EMAIL]", "[PHONE]"
        ]
        found_placeholders = [p for p in placeholders if p in response.upper()]
        if found_placeholders:
            violations.append(SafetyViolation(
                check_type="placeholder_text",
                severity="critical",
                message=f"Contains placeholder text: {', '.join(found_placeholders)}",
                action="block"
            ))
        
        # Check response length
        if len(response.strip()) < 50:
            violations.append(SafetyViolation(
                check_type="too_short",
                severity="high",
                message="Response is too short (< 50 chars)",
                action="flag"
            ))
        
        return violations

    def _fallback_qa_result(
        self, 
        response: str, 
        safety_violations: List[str] = None
    ) -> QAResult:
        """
        Fallback QA evaluation using simple heuristics.
        
        Args:
            response: Generated email response
            safety_violations: List of safety violation messages
            
        Returns:
            QAResult based on basic checks
        """
        if safety_violations is None:
            safety_violations = []
            
        issues = []
        suggestions = []
        score = 8.0
        tone_parts = []

        response_lower = response.lower()

        # Check for escalation
        escalation_keywords = [
            "escalat", "specialist", "billing team", "technical team",
            "product team", "manual review"
        ]
        is_escalation = any(keyword in response_lower for keyword in escalation_keywords)

        if is_escalation:
            return QAResult(
                approved=True,
                quality_score=9.0,
                issues=[],
                suggestions=[],
                tone_assessment="Professional escalation response",
                reasoning="Escalation to appropriate team is acceptable",
                safety_violations=safety_violations,
                should_block=False,
                requires_redaction=False
            )

        # Length checks
        if len(response) < 80:
            issues.append("Response too brief")
            suggestions.append("Expand with specific steps or information")
            score -= 3.0
        elif len(response) > 3000:
            issues.append("Response very long")
            score -= 1.5

        # Greeting check
        has_greeting = any(
            response.startswith(word) or f"\n{word}" in response[:100]
            for word in ["Dear", "Hi", "Hello"]
        )
        if not has_greeting:
            issues.append("Missing greeting")
            score -= 0.8

        # Empathy check
        empathy_words = ["understand", "appreciate", "apologize", "thank you for"]
        has_empathy = any(word in response_lower for word in empathy_words)
        if not has_empathy and len(response) > 100:
            issues.append("Could show more empathy")
            score -= 0.7

        # Required closing
        if "hope you have a great day" not in response_lower:
            issues.append("Missing required closing")
            suggestions.append("End with 'Hope you have a great day!'")
            score -= 1.0

        # Safety violations reduce score
        if safety_violations:
            issues.extend(safety_violations)
            score -= len(safety_violations) * 2.0

        # Ensure score is in valid range
        score = max(0.0, min(10.0, score))

        tone_assessment = "Professional" if score >= 7.0 else "Needs improvement"
        reasoning = f"Basic quality check: {len(issues)} issue(s) found"

        return QAResult(
            approved=score >= 7.0 and len(safety_violations) == 0,
            quality_score=round(score, 1),
            issues=issues,
            suggestions=suggestions,
            tone_assessment=tone_assessment,
            reasoning=reasoning,
            safety_violations=safety_violations,
            should_block=False,
            requires_redaction=False
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
        Perform a quick sanity check without full evaluation.
        
        Args:
            response: Email response to check
            
        Returns:
            Dictionary with quick check results
        """
        issues = []

        # Check length
        if len(response) < 50:
            issues.append("Response is too short")

        if len(response) > 3000:
            issues.append("Response is very long")

        # Check for placeholder text
        placeholders = ["[PLACEHOLDER]", "[TODO]", "[INSERT", "XXX", "TBD"]
        for placeholder in placeholders:
            if placeholder.lower() in response.lower():
                issues.append(f"Contains placeholder text: {placeholder}")

        # Check for required closing
        if "hope you have a great day" not in response.lower():
            issues.append("Missing required closing: 'Hope you have a great day!'")

        # Check product name capitalization
        if "taskflow pro" in response.lower() and "TaskFlow Pro" not in response:
            issues.append("Product name should be capitalized: 'TaskFlow Pro'")

        return {"passed": len(issues) == 0, "issues": issues}
