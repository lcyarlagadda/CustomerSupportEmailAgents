"""LLM loader for HuggingFace models with optimization support."""
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils.config import LLM_MODEL, LLM_TEMPERATURE, USE_GPU


_pipeline_cache = None
_tokenizer_cache = None
_model_cache = None


def get_device():
    """Determine which device to use."""
    if USE_GPU == "false":
        return "cpu"
    elif USE_GPU == "true":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:  # auto
        return "cuda" if torch.cuda.is_available() else "cpu"


def load_llm_pipeline(temperature=None, max_tokens=512):
    """
    Load and cache the HuggingFace pipeline with optimizations.
    
    Args:
        temperature: Override default temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        HuggingFace text generation pipeline
    """
    global _pipeline_cache, _model_cache
    
    if _pipeline_cache is not None:
        return _pipeline_cache
    
    device = get_device()
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
    
    print(f"Loading model: {LLM_MODEL}")
    print(f"Device: {device}")
    print(f"Quantization: {'Enabled (INT8)' if use_quantization and device == 'cuda' else 'Disabled'}")
    
    # Get HuggingFace token if available
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    try:
        # Load tokenizer first to configure pad token
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=hf_token)
        
        # Set pad token to suppress warnings
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization for GPU (2-3x speedup with minimal quality loss)
        model_kwargs = {
            "token": hf_token,
            "device_map": "auto" if device == "cuda" else None,
        }
        
        if device == "cuda" and use_quantization:
            # INT8 quantization - faster inference with minimal accuracy loss
            print("Applying INT8 quantization for faster inference...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
        
        # Load pipeline with optimizations
        _pipeline_cache = pipeline(
            "text-generation",
            model=LLM_MODEL,
            tokenizer=tokenizer,
            device=device if device == "cuda" else -1,  # -1 for CPU
            model_kwargs=model_kwargs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            top_k=50,  # Add top_k for better quality/speed tradeoff
            repetition_penalty=1.1,  # Reduce repetition
            pad_token_id=tokenizer.eos_token_id,
        )
        
        print(f"Model loaded successfully on {device}")
        if device == "cuda" and use_quantization:
            print("Using INT8 quantization (2-3x faster)")
        
        return _pipeline_cache
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"\nIf using Llama models:")
        print("1. Accept license at: https://huggingface.co/{LLM_MODEL}")
        print("2. Create token at: https://huggingface.co/settings/tokens")
        print("3. Set HF_TOKEN in .env file")
        
        # If quantization fails, retry without it
        if "quantization" in str(e).lower() and use_quantization:
            print("\nQuantization failed, retrying without it...")
            os.environ["USE_QUANTIZATION"] = "false"
            _pipeline_cache = None
            return load_llm_pipeline(temperature, max_tokens)
        
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

