"""Configuration settings for the support agent system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRODUCT_DOCS_DIR = DATA_DIR / "product_docs"
VECTOR_DB_DIR = PROJECT_ROOT / "vectorstore"

# LLM Configuration (Groq API)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_TEMPERATURE = 0.7

# Embedding Configuration
# Options:
# - "all-MiniLM-L6-v2" (default, 384 dim, fast, good)
# - "BAAI/bge-small-en-v1.5" (384 dim, faster, better accuracy +15%)
# - "BAAI/bge-base-en-v1.5" (768 dim, slower, best accuracy +25%)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Upgraded for better accuracy

# Vector Store Configuration
VECTOR_DB_NAME = "taskflow_support_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 4

# Email Configuration
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL", "support@taskflowpro.com")
GMAIL_CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
GMAIL_TOKEN_FILE = PROJECT_ROOT / "token.json"
CHECK_INTERVAL_SECONDS = 60

# Support Categories
EMAIL_CATEGORIES = {
    "technical_support": "Technical issues, bugs, and errors",
    "product_inquiry": "Questions about features, how-to guides, and product capabilities",
    "billing": "Billing questions, subscription issues, and payment problems",
    "feature_request": "Suggestions for new features or improvements",
    "feedback": "General feedback, compliments, or complaints",
    "unrelated": "Emails not related to product support",
}

# Response templates directory
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "support_agent.log"

# Create necessary directories
VECTOR_DB_DIR.mkdir(exist_ok=True, parents=True)
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)


def validate_config():
    """Validate that all required configuration is present."""
    errors = []
    warnings = []

    if not PRODUCT_DOCS_DIR.exists():
        errors.append(f"Product documentation directory not found: {PRODUCT_DOCS_DIR}")

    # Check if Groq API key is set
    if not GROQ_API_KEY:
        warnings.append(
            "Groq API key not found. Set GROQ_API_KEY in .env file\n"
            "  Get free API key at: https://console.groq.com"
        )

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    return True
