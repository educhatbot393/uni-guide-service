# utils/response_cleaner.py
import re

def clean_response(response):
    """
    Removes thinking blocks (<think>...</think>) from the model response.
    
    Args:
        response (str): The raw model response
        
    Returns:
        str: The cleaned response with thinking blocks removed
    """
    # Check if thinking block exists
    if '<think>' in response and '</think>' in response:
        # Use regex to remove everything between <think> and </think> tags, including the tags
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove any leading whitespace that might remain after removing the thinking block
        cleaned = cleaned.lstrip()
        
        return cleaned
    else:
        # If no thinking block is present, return the original response
        return response