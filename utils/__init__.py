"""Utility modules for customer support automation."""
from utils.config import validate_config
from utils.email_handler import Email, GmailHandler
from utils.vector_store import VectorStoreManager, retrieve_context

__all__ = [
    "validate_config",
    "Email",
    "GmailHandler",
    "VectorStoreManager",
    "retrieve_context",
]

