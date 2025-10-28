"""Unified LLM loader - supports HuggingFace (local) and Groq (API)."""

from typing import Optional
from langchain_community.llms import HuggingFacePipeline

from utils.config import LLM_PROVIDER, GROQ_MODEL, LLM_MODEL, LLM_TEMPERATURE


def load_llm(temperature: Optional[float] = None, max_tokens: int = 512):
    """
    Load LLM based on configured provider (Groq or HuggingFace).

    Args:
        temperature: Sampling temperature (overrides default)
        max_tokens: Maximum tokens to generate

    Returns:
        LLM instance (either Groq or HuggingFacePipeline)
    """
    provider = LLM_PROVIDER.lower()
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    if provider == "groq":
        print(f"Loading Groq model: {GROQ_MODEL}")
        from utils.groq_loader import load_groq_llm

        try:
            llm = load_groq_llm(model_name=GROQ_MODEL, temperature=temp, max_tokens=max_tokens)
            return llm
        except Exception as e:
            print(f"\nGroq failed: {e}")
            print("Falling back to HuggingFace...\n")
            provider = "huggingface"

    # Default to HuggingFace
    print(f"Loading HuggingFace model: {LLM_MODEL}")
    from utils.llm_loader import load_llm_pipeline

    pipe = load_llm_pipeline(temperature=temp, max_tokens=max_tokens)
    return HuggingFacePipeline(pipeline=pipe)


def get_provider_info():
    """Get information about the current LLM provider configuration."""
    return {
        "provider": LLM_PROVIDER,
        "groq_model": GROQ_MODEL,
        "huggingface_model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
    }

