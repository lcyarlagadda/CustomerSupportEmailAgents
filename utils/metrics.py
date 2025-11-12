"""Performance metrics tracking for the support system."""

import time
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EmailMetrics:
    """Metrics for a single email processing."""
    email_id: str
    category: str
    priority: str
    
    # Timing metrics
    total_time: float = 0.0
    classification_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    qa_time: float = 0.0
    
    # Quality metrics
    qa_score: float = 0.0
    qa_approved: bool = False
    revision_count: int = 0
    
    # Retrieval metrics
    docs_retrieved: int = 0
    used_cache: bool = False
    used_reranking: bool = False
    used_hybrid: bool = False
    
    # Outcome
    status: str = ""


class MetricsTracker:
    """Track and aggregate system metrics."""
    
    def __init__(self):
        self.email_metrics: List[EmailMetrics] = []
        self.session_start = time.time()
    
    def add_email_metrics(self, metrics: EmailMetrics):
        """Add metrics for a processed email."""
        self.email_metrics.append(metrics)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.email_metrics:
            return {}
        
        total_emails = len(self.email_metrics)
        
        # Timing metrics
        avg_total_time = sum(m.total_time for m in self.email_metrics) / total_emails
        avg_retrieval_time = sum(m.retrieval_time for m in self.email_metrics) / total_emails
        avg_generation_time = sum(m.generation_time for m in self.email_metrics) / total_emails
        
        # Quality metrics
        avg_qa_score = sum(m.qa_score for m in self.email_metrics) / total_emails
        approval_rate = sum(1 for m in self.email_metrics if m.qa_approved) / total_emails
        
        # Efficiency metrics
        cache_hit_rate = sum(1 for m in self.email_metrics if m.used_cache) / total_emails
        avg_revisions = sum(m.revision_count for m in self.email_metrics) / total_emails
        
        # Status breakdown
        statuses = {}
        for m in self.email_metrics:
            statuses[m.status] = statuses.get(m.status, 0) + 1
        
        return {
            "total_emails": total_emails,
            "avg_processing_time": avg_total_time,
            "avg_retrieval_time": avg_retrieval_time,
            "avg_generation_time": avg_generation_time,
            "avg_qa_score": avg_qa_score,
            "approval_rate": approval_rate,
            "cache_hit_rate": cache_hit_rate,
            "avg_revisions": avg_revisions,
            "status_breakdown": statuses,
            "session_duration": time.time() - self.session_start,
        }
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        if not summary:
            print("\nNo metrics to display")
            return
        
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        print(f"Total Emails Processed: {summary['total_emails']}")
        print(f"Session Duration: {summary['session_duration']:.1f}s")
        print()
        print("Timing (average per email):")
        print(f"  Total Processing: {summary['avg_processing_time']:.2f}s")
        print(f"  Retrieval: {summary['avg_retrieval_time']:.2f}s")
        print(f"  Generation: {summary['avg_generation_time']:.2f}s")
        print()
        print("Quality:")
        print(f"  Average QA Score: {summary['avg_qa_score']:.1f}/10")
        print(f"  Approval Rate: {summary['approval_rate']*100:.1f}%")
        print(f"  Average Revisions: {summary['avg_revisions']:.2f}")
        print()
        print("Efficiency:")
        print(f"  Cache Hit Rate: {summary['cache_hit_rate']*100:.1f}%")
        print()
        print("Status Breakdown:")
        for status, count in summary['status_breakdown'].items():
            print(f"  {status}: {count}")
        print("="*70 + "\n")


# Global tracker instance
_tracker = MetricsTracker()

def get_tracker() -> MetricsTracker:
    """Get the global metrics tracker."""
    return _tracker

def reset_tracker():
    """Reset the global metrics tracker."""
    global _tracker
    _tracker = MetricsTracker()

