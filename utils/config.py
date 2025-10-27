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

# LLM Configuration - Using HuggingFace (free, works on Colab)
LLM_PROVIDER = "huggingface"
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-1B-Instruct")  # Llama 3.2 1B for Colab
LLM_TEMPERATURE = 0.7
USE_GPU = os.getenv("USE_GPU", "auto")  # auto, true, false

# Embedding Configuration - Using local HuggingFace embeddings (free)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient local embeddings

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
    "unrelated": "Emails not related to product support"
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
    
    # Check if transformers is installed
    try:
        import transformers
    except ImportError:
        errors.append("transformers package not found. Run: pip install transformers torch")
    
    # Warn if HuggingFace token not set (needed for gated models)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token and "meta-llama" in LLM_MODEL.lower():
        warnings.append(
            "HuggingFace token not found. For gated models (Llama), set HF_TOKEN in .env\n"
            "  Get token at: https://huggingface.co/settings/tokens"
        )
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    
    return True


if __name__ == "__main__":
    try:
        validate_config()
        print("Configuration validated successfully")
        print(f"LLM Provider: {LLM_PROVIDER}")
        print(f"LLM Model: {LLM_MODEL}")
        print(f"Product docs directory: {PRODUCT_DOCS_DIR}")
        print(f"Vector DB directory: {VECTOR_DB_DIR}")
    except ValueError as e:
        print(f"Configuration validation failed:\n{e}")

