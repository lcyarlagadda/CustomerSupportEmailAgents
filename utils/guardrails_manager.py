"""
Guardrails Manager for Safety and Validation

This module provides safety checks for the support agent system:
1. PII Detection - Detects and flags sensitive information
2. Toxic Language - Identifies abusive or inappropriate content
3. Topic Restriction - Ensures responses stay on-topic
4. Content Moderation - Additional safety checks

Usage:
    from utils.guardrails_manager import GuardrailsManager
    
    guardrails = GuardrailsManager()
    
    # Validate incoming email
    email_check = guardrails.validate_incoming_email(email_body)
    
    # Validate outgoing response
    response_check = guardrails.validate_response(response_text)
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class GuardrailViolation:
    """Represents a guardrail violation."""
    check_type: str  # "pii", "toxic", "off_topic", "moderation"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    field: Optional[str] = None  # Which field has the issue
    action: str = "flag"  # "flag", "block", "redact"


class GuardrailsManager:
    """
    Comprehensive safety and validation manager.
    
    Provides multiple layers of safety checks:
    - PII detection (emails, phones, SSNs, credit cards)
    - Toxic language detection
    - Topic restriction
    - Content moderation
    """
    
    def __init__(
        self,
        enable_pii_detection: bool = True,
        enable_toxic_detection: bool = True,
        enable_topic_restriction: bool = True,
        toxic_threshold: float = 0.7,
        valid_topics: List[str] = None
    ):
        """
        Initialize guardrails manager.
        
        Args:
            enable_pii_detection: Enable PII detection
            enable_toxic_detection: Enable toxic language detection
            enable_topic_restriction: Enable topic restriction
            toxic_threshold: Threshold for toxic language (0-1, higher = stricter)
            valid_topics: List of valid topics
        """
        self.enable_pii_detection = enable_pii_detection
        self.enable_toxic_detection = enable_toxic_detection
        self.enable_topic_restriction = enable_topic_restriction
        self.toxic_threshold = toxic_threshold
        
        self.valid_topics = valid_topics or [
            "billing",
            "technical support",
            "product inquiry",
            "integration",
            "account management",
            "feature request",
            "feedback"
        ]
        
        # Try to initialize Guardrails AI
        self.guardrails_available = False
        try:
            from guardrails import Guard
            from guardrails.hub import DetectPII, ToxicLanguage, RestrictToTopic
            
            self.Guard = Guard
            self.DetectPII = DetectPII
            self.ToxicLanguage = ToxicLanguage
            self.RestrictToTopic = RestrictToTopic
            self.guardrails_available = True
        except ImportError:
            # Fallback to simple regex-based checks
            pass
        
        # Initialize PII regex patterns (fallback)
        self.pii_patterns = {
            "EMAIL_ADDRESS": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "PHONE_NUMBER": re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'),
            "SSN": re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            "CREDIT_CARD": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            "IP_ADDRESS": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        }
        
        # Toxic keywords (simple fallback)
        self.toxic_keywords = [
            "stupid", "idiot", "moron", "dumb", "hate", "worst",
            "useless", "garbage", "trash", "pathetic", "incompetent"
        ]
    
    def validate_incoming_email(self, email_body: str, email_subject: str = "") -> Dict[str, Any]:
        """
        Validate incoming customer email for safety issues.
        
        Args:
            email_body: Email body text
            email_subject: Email subject text
            
        Returns:
            Dict with validation results:
            {
                "is_safe": bool,
                "violations": List[GuardrailViolation],
                "requires_human_review": bool,
                "warnings": List[str]
            }
        """
        violations = []
        warnings = []
        
        # Check for PII in incoming email (informational only)
        if self.enable_pii_detection:
            pii_violations = self._check_pii(email_body, field="email_body")
            if pii_violations:
                # PII in incoming email is informational (customer can share their own info)
                warnings.extend([f"Customer shared {v.check_type}: {v.message}" for v in pii_violations])
        
        # Check for toxic language from customer
        if self.enable_toxic_detection:
            toxic_violations = self._check_toxic_language(email_body + " " + email_subject)
            if toxic_violations:
                violations.extend(toxic_violations)
        
        # Determine if human review is needed
        requires_human_review = any(
            v.severity in ["high", "critical"] for v in violations
        )
        
        return {
            "is_safe": len(violations) == 0,
            "violations": violations,
            "requires_human_review": requires_human_review,
            "warnings": warnings
        }
    
    def validate_response(
        self,
        response_text: str,
        email_context: str = "",
        category: str = ""
    ) -> Dict[str, Any]:
        """
        Validate outgoing response for safety issues.
        
        Args:
            response_text: The generated response
            email_context: Original email for context
            category: Email category
            
        Returns:
            Dict with validation results:
            {
                "is_safe": bool,
                "violations": List[GuardrailViolation],
                "should_block": bool,
                "redacted_response": Optional[str]
            }
        """
        violations = []
        redacted_response = response_text
        
        # Check for PII leakage in response (CRITICAL)
        if self.enable_pii_detection:
            pii_violations = self._check_pii(response_text, field="response")
            for violation in pii_violations:
                # PII in response is critical - we should redact
                violation.severity = "critical"
                violation.action = "redact"
                violations.append(violation)
                
                # Redact PII
                redacted_response = self._redact_pii(redacted_response)
        
        # Check for toxic language in response (CRITICAL)
        if self.enable_toxic_detection:
            toxic_violations = self._check_toxic_language(response_text)
            for violation in toxic_violations:
                violation.severity = "critical"
                violation.action = "block"
                violations.append(violation)
        
        # Check if response is on-topic
        if self.enable_topic_restriction and category:
            topic_violations = self._check_topic_restriction(response_text, category)
            violations.extend(topic_violations)
        
        # Additional safety checks
        additional_violations = self._additional_safety_checks(response_text)
        violations.extend(additional_violations)
        
        # Determine if we should block the response
        should_block = any(
            v.action == "block" or v.severity == "critical" for v in violations
        )
        
        return {
            "is_safe": len(violations) == 0,
            "violations": violations,
            "should_block": should_block,
            "redacted_response": redacted_response if redacted_response != response_text else None
        }
    
    def _check_pii(self, text: str, field: str = "text") -> List[GuardrailViolation]:
        """Check for PII in text."""
        violations = []
        
        if self.guardrails_available:
            # Use Guardrails AI if available
            try:
                from guardrails.hub import DetectPII
                
                pii_entities = ["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD"]
                detector = DetectPII(pii_entities=pii_entities)
                
                # Note: Actual Guardrails API may differ, this is a simplified version
                # You may need to adjust based on the actual API
                
            except Exception:
                pass
        
        # Fallback: Regex-based detection
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                violations.append(GuardrailViolation(
                    check_type=f"pii_{pii_type.lower()}",
                    severity="high" if pii_type in ["SSN", "CREDIT_CARD"] else "medium",
                    message=f"Detected {pii_type}: {len(matches)} occurrence(s)",
                    field=field,
                    action="redact"
                ))
        
        return violations
    
    def _check_toxic_language(self, text: str) -> List[GuardrailViolation]:
        """Check for toxic/abusive language."""
        violations = []
        text_lower = text.lower()
        
        # Simple keyword-based check (fallback)
        found_keywords = [kw for kw in self.toxic_keywords if kw in text_lower]
        
        if found_keywords:
            violations.append(GuardrailViolation(
                check_type="toxic_language",
                severity="high",
                message=f"Potentially toxic language detected: {', '.join(found_keywords[:3])}",
                action="block"
            ))
        
        # Check for excessive caps (shouting)
        words = text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if len(caps_words) > len(words) * 0.3:  # More than 30% caps
            violations.append(GuardrailViolation(
                check_type="excessive_caps",
                severity="medium",
                message="Excessive use of capital letters (perceived as shouting)",
                action="flag"
            ))
        
        # Check for excessive punctuation
        if text.count('!') > 3 or text.count('?') > 3:
            violations.append(GuardrailViolation(
                check_type="excessive_punctuation",
                severity="low",
                message="Excessive use of punctuation",
                action="flag"
            ))
        
        return violations
    
    def _check_topic_restriction(self, response: str, category: str) -> List[GuardrailViolation]:
        """Check if response stays on-topic."""
        violations = []
        response_lower = response.lower()
        
        # Map categories to expected keywords
        category_keywords = {
            "billing": ["price", "payment", "invoice", "subscription", "refund", "charge"],
            "technical_support": ["error", "issue", "problem", "fix", "troubleshoot", "configure"],
            "product_inquiry": ["feature", "functionality", "how to", "capability", "available"],
            "integration": ["api", "integration", "connect", "webhook", "endpoint"],
            "account_management": ["account", "login", "password", "profile", "settings"],
        }
        
        # Get expected keywords for this category
        expected_keywords = category_keywords.get(category, [])
        
        if expected_keywords:
            # Check if response contains at least some relevant keywords
            found_keywords = sum(1 for kw in expected_keywords if kw in response_lower)
            
            if found_keywords == 0 and len(response) > 100:
                violations.append(GuardrailViolation(
                    check_type="off_topic",
                    severity="medium",
                    message=f"Response may be off-topic for category: {category}",
                    action="flag"
                ))
        
        return violations
    
    def _additional_safety_checks(self, response: str) -> List[GuardrailViolation]:
        """Additional safety and quality checks."""
        violations = []
        
        # Check for placeholder text
        placeholders = ["[INSERT", "[TODO", "{{", "[NAME]", "[COMPANY]", "XXX"]
        found_placeholders = [p for p in placeholders if p in response.upper()]
        if found_placeholders:
            violations.append(GuardrailViolation(
                check_type="placeholder_text",
                severity="critical",
                message=f"Response contains placeholder text: {found_placeholders}",
                action="block"
            ))
        
        # Check response length
        if len(response) < 50:
            violations.append(GuardrailViolation(
                check_type="too_short",
                severity="medium",
                message="Response is very short (< 50 chars)",
                action="flag"
            ))
        
        if len(response) > 2000:
            violations.append(GuardrailViolation(
                check_type="too_long",
                severity="low",
                message="Response is very long (> 2000 chars)",
                action="flag"
            ))
        
        # Check for proper structure
        if "TaskFlow Pro Team" not in response:
            violations.append(GuardrailViolation(
                check_type="missing_signature",
                severity="low",
                message="Response missing proper signature",
                action="flag"
            ))
        
        return violations
    
    def _redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        redacted = text
        
        # Redact email addresses
        redacted = self.pii_patterns["EMAIL_ADDRESS"].sub("[EMAIL REDACTED]", redacted)
        
        # Redact phone numbers
        redacted = self.pii_patterns["PHONE_NUMBER"].sub("[PHONE REDACTED]", redacted)
        
        # Redact SSNs
        redacted = self.pii_patterns["SSN"].sub("[SSN REDACTED]", redacted)
        
        # Redact credit cards
        redacted = self.pii_patterns["CREDIT_CARD"].sub("[CARD REDACTED]", redacted)
        
        return redacted
    
    def get_summary(self, violations: List[GuardrailViolation]) -> Dict[str, Any]:
        """Get summary of violations."""
        if not violations:
            return {
                "total_violations": 0,
                "by_severity": {},
                "by_type": {},
                "requires_action": False
            }
        
        by_severity = {}
        by_type = {}
        
        for v in violations:
            by_severity[v.severity] = by_severity.get(v.severity, 0) + 1
            by_type[v.check_type] = by_type.get(v.check_type, 0) + 1
        
        requires_action = any(v.action in ["block", "redact"] for v in violations)
        
        return {
            "total_violations": len(violations),
            "by_severity": by_severity,
            "by_type": by_type,
            "requires_action": requires_action
        }


# Global instance
_guardrails_manager = None


def get_guardrails_manager() -> GuardrailsManager:
    """Get or create the global guardrails manager instance."""
    global _guardrails_manager
    if _guardrails_manager is None:
        _guardrails_manager = GuardrailsManager(
            enable_pii_detection=True,
            enable_toxic_detection=True,
            enable_topic_restriction=True,
            toxic_threshold=0.7
        )
    return _guardrails_manager

