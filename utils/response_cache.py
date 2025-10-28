"""Response caching system for FAQ-style emails."""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta


class ResponseCache:
    """Cache system for storing and retrieving common email responses."""

    def __init__(self, cache_file: str = "cache/response_cache.json", ttl_hours: int = 168):
        """
        Initialize response cache.

        Args:
            cache_file: Path to cache file
            ttl_hours: Time-to-live for cache entries in hours (default: 168 = 1 week)
        """
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _generate_key(self, email_subject: str, email_body: str, category: str) -> str:
        """
        Generate a cache key from email content.

        Args:
            email_subject: Email subject
            email_body: Email body (first 500 chars)
            category: Email category

        Returns:
            Cache key (hash)
        """
        # Use first 500 chars of body for key generation
        content = f"{category}:{email_subject}:{email_body[:500]}".lower().strip()
        # Normalize whitespace
        content = " ".join(content.split())
        # Generate hash
        return hashlib.md5(content.encode()).hexdigest()

    def _is_expired(self, timestamp: str) -> bool:
        """Check if a cache entry is expired."""
        try:
            cached_time = datetime.fromisoformat(timestamp)
            return datetime.now() - cached_time > self.ttl
        except:
            return True

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two texts are similar using simple word overlap.

        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (0-1)

        Returns:
            True if texts are similar enough
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1 & words2
        union = words1 | words2

        similarity = len(intersection) / len(union)
        return similarity >= threshold

    def get(
        self, email_subject: str, email_body: str, category: str, use_fuzzy: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available.

        Args:
            email_subject: Email subject
            email_body: Email body
            category: Email category
            use_fuzzy: Try fuzzy matching if exact match not found

        Returns:
            Cached response dict or None if not found/expired
        """
        # Try exact match first
        key = self._generate_key(email_subject, email_body, category)

        if key in self.cache:
            entry = self.cache[key]
            if not self._is_expired(entry["timestamp"]):
                print(f"✓ Cache hit (exact match)")
                entry["cache_hit_type"] = "exact"
                return entry
            else:
                # Remove expired entry
                del self.cache[key]
                self._save_cache()

        # Try fuzzy matching for similar emails
        if use_fuzzy:
            for cached_key, entry in list(self.cache.items()):
                # Skip expired entries
                if self._is_expired(entry["timestamp"]):
                    del self.cache[cached_key]
                    continue

                # Check if category matches and content is similar
                if entry["category"] == category:
                    if self._is_similar(
                        email_subject + " " + email_body, entry["subject"] + " " + entry["body"]
                    ):
                        print(f"Cache hit (fuzzy match)")
                        entry["cache_hit_type"] = "fuzzy"
                        return entry

        return None

    def set(
        self,
        email_subject: str,
        email_body: str,
        category: str,
        response: str,
        qa_score: float = 0.0,
        metadata: Optional[Dict] = None,
    ):
        """
        Store a response in cache.

        Args:
            email_subject: Email subject
            email_body: Email body
            category: Email category
            response: Generated response
            qa_score: Quality score
            metadata: Additional metadata
        """
        key = self._generate_key(email_subject, email_body, category)

        self.cache[key] = {
            "subject": email_subject,
            "body": email_body[:500],  # Store truncated body
            "category": category,
            "response": response,
            "qa_score": qa_score,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self._save_cache()
        print(f"✓ Response cached (key: {key[:8]}...)")

    def clear_expired(self):
        """Remove all expired entries from cache."""
        initial_size = len(self.cache)
        self.cache = {k: v for k, v in self.cache.items() if not self._is_expired(v["timestamp"])}
        removed = initial_size - len(self.cache)
        if removed > 0:
            self._save_cache()
            print(f"Cleared {removed} expired cache entries")

    def clear_all(self):
        """Clear entire cache."""
        self.cache = {}
        self._save_cache()
        print("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = len(self.cache)
        expired = sum(1 for v in self.cache.values() if self._is_expired(v["timestamp"]))

        # Category breakdown
        categories = {}
        for entry in self.cache.values():
            cat = entry.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_entries": total,
            "active_entries": total - expired,
            "expired_entries": expired,
            "by_category": categories,
            "cache_file": str(self.cache_file),
            "ttl_hours": self.ttl.total_seconds() / 3600,
        }
