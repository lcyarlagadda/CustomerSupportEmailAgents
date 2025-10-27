"""LLM loader for HuggingFace models."""
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from utils.config import LLM_MODEL, LLM_TEMPERATURE, USE_GPU


_pipeline_cache = None
_tokenizer_cache = None


def get_device():
    """Determine which device to use."""
    if USE_GPU == "false":
        return "cpu"
    elif USE_GPU == "true":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:  # auto
        return "cuda" if torch.cuda.is_available() else "cpu"


def load_llm_pipeline(temperature=None):
    """
    Load and cache the HuggingFace pipeline.
    
    Args:
        temperature: Override default temperature
        
    Returns:
        HuggingFace text generation pipeline
    """
    global _pipeline_cache
    
    if _pipeline_cache is not None:
        return _pipeline_cache
    
    device = get_device()
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    
    print(f"Loading model: {LLM_MODEL}")
    print(f"Device: {device}")
    
    # Get HuggingFace token if available
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    try:
        # Load pipeline with appropriate settings for Colab
        _pipeline_cache = pipeline(
            "text-generation",
            model=LLM_MODEL,
            device=device if device == "cuda" else -1,  # -1 for CPU
            token=hf_token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            max_new_tokens=512,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
        )
        
        print(f"Model loaded successfully on {device}")
        return _pipeline_cache
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"\nIf using Llama models:")
        print("1. Accept license at: https://huggingface.co/{LLM_MODEL}")
        print("2. Create token at: https://huggingface.co/settings/tokens")
        print("3. Set HF_TOKEN in .env file")
        raise


def load_tokenizer():
    """Load and cache the tokenizer."""
    global _tokenizer_cache
    
    if _tokenizer_cache is not None:
        return _tokenizer_cache
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    _tokenizer_cache = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        token=hf_token
    )
    
    return _tokenizer_cache


def generate_text(prompt, max_tokens=512, temperature=None):
    """
    Generate text from a prompt.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    pipe = load_llm_pipeline(temperature)
    
    # Generate with appropriate parameters
    result = pipe(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature or LLM_TEMPERATURE,
        do_sample=True,
        top_p=0.9,
        return_full_text=False
    )
    
    return result[0]["generated_text"]


if __name__ == "__main__":
    # Test the loader
    print("Testing LLM loader...")
    print("-" * 50)
    
    try:
        pipe = load_llm_pipeline()
        print("\nTesting generation...")
        
        test_prompt = "What is Python?"
        result = generate_text(test_prompt, max_tokens=50)
        
        print(f"\nPrompt: {test_prompt}")
        print(f"Response: {result}")
        print("\nSuccess!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

