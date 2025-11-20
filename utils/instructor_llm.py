"""
Instructor-wrapped LLM loader for structured outputs with Pydantic models.

This module provides LLM instances that automatically return validated Pydantic models
instead of raw text, ensuring type safety and validation.
"""

import instructor
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
from groq import Groq
from utils.groq_loader import get_groq_api_key
from utils.config import GROQ_MODEL, LLM_TEMPERATURE

T = TypeVar('T', bound=BaseModel)

# Cache for instructor clients
_instructor_cache = {}


def get_instructor_client(
    temperature: Optional[float] = None,
    max_tokens: int = 512,
    model: Optional[str] = None
):
    """
    Get an instructor-wrapped Groq client for structured outputs.
    
    Args:
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        model: Model name (defaults to GROQ_MODEL from config)
        
    Returns:
        Instructor client that can return structured Pydantic models
    """
    model_name = model or GROQ_MODEL
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    
    # Create cache key
    cache_key = f"{model_name}_{temp}_{max_tokens}"
    
    if cache_key in _instructor_cache:
        return _instructor_cache[cache_key]
    
    # Create raw Groq client (not LangChain wrapper)
    api_key = get_groq_api_key()
    base_client = Groq(api_key=api_key)
    
    # Wrap with instructor for structured outputs
    # Instructor automatically handles JSON mode and validation
    client = instructor.from_groq(
        base_client,
        mode=instructor.Mode.JSON
    )
    
    # Cache the client
    _instructor_cache[cache_key] = client
    
    return client


def get_structured_response(
    prompt: str,
    response_model: Type[T],
    temperature: Optional[float] = None,
    max_tokens: int = 512,
    model: Optional[str] = None
) -> T:
    """
    Get a structured response from the LLM that conforms to a Pydantic model.
    
    Args:
        prompt: The prompt to send to the LLM
        response_model: Pydantic model class for the response structure
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        model: Model name (optional)
        
    Returns:
        Validated Pydantic model instance
        
    Example:
        ```python
        from pydantic import BaseModel
        
        class Classification(BaseModel):
            category: str
            confidence: float
            
        result = get_structured_response(
            prompt="Classify this email: ...",
            response_model=Classification,
            temperature=0.3
        )
        
        print(result.category)  # Guaranteed to be a string
        print(result.confidence)  # Guaranteed to be a float
        ```
    """
    client = get_instructor_client(temperature=temperature, max_tokens=max_tokens, model=model)
    
    # Instructor automatically handles the structured output
    response = client.chat.completions.create(
        model=model or GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=response_model,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens
    )
    
    return response


def get_structured_response_with_retries(
    prompt: str,
    response_model: Type[T],
    temperature: Optional[float] = None,
    max_tokens: int = 512,
    max_retries: int = 3
) -> T:
    """
    Get a structured response with automatic retries on validation errors.
    
    Args:
        prompt: The prompt to send to the LLM
        response_model: Pydantic model class for the response structure
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retry attempts
        
    Returns:
        Validated Pydantic model instance
        
    Raises:
        ValueError: If all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return get_structured_response(
                prompt=prompt,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Increase temperature slightly for retry
                temperature = (temperature or LLM_TEMPERATURE) + 0.1
                continue
            else:
                raise ValueError(f"Failed to get structured response after {max_retries} attempts: {e}")
    
    raise ValueError(f"Failed to get structured response: {last_error}")

