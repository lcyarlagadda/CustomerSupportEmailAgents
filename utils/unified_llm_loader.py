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
            llm = load_groq_llm(
                model_name=GROQ_MODEL,
                temperature=temp,
                max_tokens=max_tokens
            )
            return llm
        except Exception as e:
            print(f"\n⚠️  Groq failed: {e}")
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
        'provider': LLM_PROVIDER,
        'groq_model': GROQ_MODEL,
        'huggingface_model': LLM_MODEL,
        'temperature': LLM_TEMPERATURE
    }


def print_provider_info():
    """Print current LLM provider configuration."""
    info = get_provider_info()
    
    print("\n" + "=" * 60)
    print("LLM CONFIGURATION")
    print("=" * 60)
    print(f"Provider: {info['provider'].upper()}")
    
    if info['provider'].lower() == 'groq':
        print(f"Model: {info['groq_model']}")
        print("\n✓ Using Groq API (blazing fast, FREE 14.4k requests/day)")
        print("  - Speed: 1-2 seconds per email")
        print("  - 300-500 tokens/second")
        print("  - Free tier: 14,400 requests/day")
    else:
        print(f"Model: {info['huggingface_model']}")
        print("\n✓ Using HuggingFace (local, unlimited, FREE)")
        print("  - Direct model access")
        print("  - INT8 quantization support")
        print("  - Works on GPU/CPU")
    
    print(f"\nTemperature: {info['temperature']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Testing Unified LLM Loader...")
    print("-" * 60)
    
    # Print current configuration
    print_provider_info()
    
    # Test loading
    print("\nAttempting to load LLM...")
    try:
        llm = load_llm(temperature=0.7, max_tokens=50)
        print("✓ LLM loaded successfully\n")
        
        # Test generation
        print("Testing generation...")
        response = llm.invoke("What is 2+2? Answer briefly.")
        
        # Handle different response types
        if hasattr(response, 'content'):
            print(f"Response: {response.content}\n")
        else:
            print(f"Response: {response}\n")
        
        print("✓ Unified LLM loader working!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-" * 60)

