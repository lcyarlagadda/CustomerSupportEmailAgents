"""Simplified LLM loader - Uses Groq API exclusively."""

from typing import Optional
from utils.config import GROQ_MODEL, LLM_TEMPERATURE


def load_llm(temperature: Optional[float] = None, max_tokens: int = 512):
    """
    Load Groq LLM.

    Args:
        temperature: Sampling temperature (overrides default)
        max_tokens: Maximum tokens to generate

    Returns:
        Groq LLM instance
    """
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    
    from utils.groq_loader import load_groq_llm
    
    llm = load_groq_llm(model_name=GROQ_MODEL, temperature=temp, max_tokens=max_tokens)
    return llm


def get_provider_info():
    """Get information about the current LLM provider configuration."""
    return {
        "provider": "groq",
        "model": GROQ_MODEL,
        "temperature": LLM_TEMPERATURE,
    }

