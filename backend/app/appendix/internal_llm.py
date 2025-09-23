"""
Mock internal LLM module for testing
"""

def get_llm_response(prompt: str, context: str = "") -> str:
    """Mock LLM response function"""
    return f"Mock response for prompt: {prompt[:50]}..."