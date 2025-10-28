"""Groq API loader - blazing fast LLM inference with generous free tier."""

import os
from typing import Optional
from langchain_groq import ChatGroq

from utils.config import LLM_TEMPERATURE


_groq_cache = {}


def check_groq_api_key() -> bool:
    """
    Check if Groq API key is set.

    Returns:
        True if API key exists, False otherwise
    """
    api_key = os.getenv("GROQ_API_KEY")
    return api_key is not None and len(api_key) > 0


def get_groq_api_key() -> str:
    """
    Get Groq API key from environment.

    Returns:
        API key

    Raises:
        ValueError if API key not found
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found!\n\n")

    return api_key


def load_groq_llm(model_name: str, temperature: Optional[float] = None, max_tokens: int = 512):
    """
    Load Groq LLM.

    Args:
        model_name: Groq model name (e.g., 'llama-3.2-3b-preview', 'mixtral-8x7b-32768')
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        ChatGroq instance

    """
    global _groq_cache

    # Create cache key
    cache_key = f"{model_name}_{temperature}_{max_tokens}"

    if cache_key in _groq_cache:
        return _groq_cache[cache_key]

    # Get API key
    api_key = get_groq_api_key()

    # Set temperature
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    # Create Groq instance
    llm = ChatGroq(
        model=model_name,
        temperature=temp,
        max_tokens=max_tokens,
        groq_api_key=api_key,
        model_kwargs={
            "top_p": 0.9,
            "frequency_penalty": 0.1,  # Reduce repetition
        },
    )

    # Cache the instance
    _groq_cache[cache_key] = llm

    print(f" Groq model '{model_name}' loaded successfully")
    print(f"  Speed: 300-500 tokens/second")
    print(f"  Free tier: 14,400 requests/day")

    return llm