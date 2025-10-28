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
        raise ValueError(
            "GROQ_API_KEY not found!\n\n"
            "Get your FREE API key:\n"
            "1. Visit: https://console.groq.com/keys\n"
            "2. Sign up (free)\n"
            "3. Create API key\n"
            "4. Add to .env: GROQ_API_KEY=your_key_here\n\n"
            "Free Tier: 14,400 requests/day (plenty for most use cases!)"
        )
    
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
    
    Available Models (Current as of Dec 2024):
        - llama-3.3-70b-versatile - Llama 3.3 70B (best quality)
        - llama-3.1-70b-versatile - Llama 3.1 70B (excellent)
        - llama-3.1-8b-instant - Llama 3.1 8B (fast, recommended)
        - mixtral-8x7b-32768 - Mixtral 8x7B (excellent, 32k context)
        - gemma2-9b-it - Gemma 2 9B (good alternative)
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
        }
    )
    
    # Cache the instance
    _groq_cache[cache_key] = llm
    
    print(f"✓ Groq model '{model_name}' loaded successfully")
    print(f"  Speed: 300-500 tokens/second (blazing fast!) ⚡")
    print(f"  Free tier: 14,400 requests/day")
    
    return llm


def list_recommended_models():
    """Print recommended Groq models."""
    recommendations = {
        'llama-3.1-8b-instant': {
            'speed': '⚡⚡⚡⚡ Fastest',
            'quality': 'Excellent',
            'recommended': True,
            'notes': 'Best balance - recommended for most use cases (DEFAULT)'
        },
        'llama-3.3-70b-versatile': {
            'speed': '⚡⚡ Fast',
            'quality': 'Best',
            'recommended': True,
            'notes': 'Highest quality, latest model'
        },
        'llama-3.1-70b-versatile': {
            'speed': '⚡⚡ Fast',
            'quality': 'Excellent',
            'recommended': True,
            'notes': 'Very high quality, stable'
        },
        'mixtral-8x7b-32768': {
            'speed': '⚡⚡ Fast',
            'quality': 'Excellent',
            'recommended': True,
            'notes': 'Great for long emails, 32k context'
        },
        'gemma2-9b-it': {
            'speed': '⚡⚡⚡ Very Fast',
            'quality': 'Very Good',
            'recommended': False,
            'notes': 'Google model, good alternative'
        }
    }
    
    print("\n" + "=" * 70)
    print("RECOMMENDED GROQ MODELS (All FREE up to 14.4k requests/day)")
    print("=" * 70)
    
    for model, info in recommendations.items():
        if info['recommended']:
            print(f"\n✓ {model}")
            print(f"  Speed: {info['speed']} | Quality: {info['quality']}")
            print(f"  {info['notes']}")
    
    print("\n" + "=" * 70)
    print("\nTo use Groq:")
    print("1. Get FREE API key: https://console.groq.com/keys")
    print("2. Add to .env: GROQ_API_KEY=your_key_here")
    print("3. Set: LLM_PROVIDER=groq")
    print("4. Set: GROQ_MODEL=llama-3.1-8b-instant  (default)")
    print("\nProcessing speed: 1-2 seconds per email! ⚡")
    print("=" * 70 + "\n")


def get_usage_info():
    """Get information about Groq free tier usage."""
    return {
        'free_tier_limit': '14,400 requests/day',
        'rate_limit': '6,000 requests/minute',
        'estimated_emails': '~432,000 emails/month (free)',
        'speed': '300-500 tokens/second',
        'cost_after_free': '$0.05-0.27 per million tokens',
        'signup_url': 'https://console.groq.com'
    }


if __name__ == "__main__":
    print("Testing Groq Integration...")
    print("-" * 60)
    
    # Check if API key is set
    print("\n1. Checking Groq API key...")
    if check_groq_api_key():
        print("✓ GROQ_API_KEY found")
        
        # Show usage info
        print("\n2. Groq Free Tier Info:")
        usage = get_usage_info()
        for key, value in usage.items():
            print(f"  {key}: {value}")
        
        # Show recommended models
        print("\n3. Recommended models:")
        list_recommended_models()
        
        # Try loading a model (if API key is valid)
        print("\n4. Testing model loading...")
        try:
            llm = load_groq_llm("llama-3.2-3b-preview", temperature=0.7, max_tokens=50)
            
            # Test generation
            print("\n5. Testing generation...")
            response = llm.invoke("What is 2+2? Answer in one sentence.")
            print(f"Response: {response.content}")
            print("\n✓ Groq integration working!")
            
        except Exception as e:
            print(f"Error: {e}")
            if "API key" in str(e):
                print("\nPlease set a valid GROQ_API_KEY in .env")
    else:
        print("✗ GROQ_API_KEY not found")
        print("\nTo get started with Groq (FREE):")
        print("1. Visit: https://console.groq.com/keys")
        print("2. Sign up (free, no credit card required)")
        print("3. Create API key")
        print("4. Add to .env: GROQ_API_KEY=your_key_here")
        print("\nFree tier: 14,400 requests/day (plenty!)")
    
    print("\n" + "-" * 60)

