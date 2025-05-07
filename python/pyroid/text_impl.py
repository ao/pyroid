"""
Pyroid Text Implementation
======================

This module provides Python implementations of the text functions.

Functions:
    reverse: Reverse a string
    base64_encode: Encode a string to base64
    base64_decode: Decode a base64 string
    split: Split a string by a delimiter
    join: Join a list of strings with a delimiter
    replace: Replace a substring in a string
    regex_replace: Replace a regex pattern in a string
    to_uppercase: Convert a string to uppercase
    to_lowercase: Convert a string to lowercase
    tokenize: Tokenize a string into words
    ngrams: Generate n-grams from a string
"""

import base64
import re
from typing import List, Union

def reverse(text: str) -> str:
    """
    Reverse a string.
    
    Args:
        text: The string to reverse
        
    Returns:
        The reversed string
    """
    return text[::-1]

def base64_encode(text: str) -> str:
    """
    Encode a string to base64.
    
    Args:
        text: The string to encode
        
    Returns:
        The base64-encoded string
    """
    return base64.b64encode(text.encode()).decode()

def base64_decode(text: str) -> str:
    """
    Decode a base64 string.
    
    Args:
        text: The base64-encoded string to decode
        
    Returns:
        The decoded string
    """
    return base64.b64decode(text.encode()).decode()

def split(text: str, delimiter: str) -> List[str]:
    """
    Split a string by a delimiter.
    
    Args:
        text: The string to split
        delimiter: The delimiter to split by
        
    Returns:
        The list of substrings
    """
    return text.split(delimiter)

def join(strings: List[str], delimiter: str) -> str:
    """
    Join a list of strings with a delimiter.
    
    Args:
        strings: The strings to join
        delimiter: The delimiter to join with
        
    Returns:
        The joined string
    """
    return delimiter.join(strings)

def replace(text: str, old: str, new: str) -> str:
    """
    Replace a substring in a string.
    
    Args:
        text: The string to modify
        old: The substring to replace
        new: The replacement substring
        
    Returns:
        The modified string
    """
    return text.replace(old, new)

def regex_replace(text: str, pattern: str, replacement: str) -> str:
    """
    Replace a regex pattern in a string.
    
    Args:
        text: The string to modify
        pattern: The regex pattern to replace
        replacement: The replacement string
        
    Returns:
        The modified string
    """
    return re.sub(pattern, replacement, text)

def to_uppercase(text: str) -> str:
    """
    Convert a string to uppercase.
    
    Args:
        text: The string to convert
        
    Returns:
        The uppercase string
    """
    return text.upper()

def to_lowercase(text: str) -> str:
    """
    Convert a string to lowercase.
    
    Args:
        text: The string to convert
        
    Returns:
        The lowercase string
    """
    return text.lower()

def tokenize(text: str, lowercase: bool = True, remove_punct: bool = True) -> List[str]:
    """
    Tokenize a string into words.
    
    Args:
        text: The string to tokenize
        lowercase: Whether to convert to lowercase
        remove_punct: Whether to remove punctuation
        
    Returns:
        The list of tokens
    """
    if lowercase:
        text = text.lower()
    
    if remove_punct:
        text = re.sub(r'[^\w\s]', '', text)
    
    return text.split()

def ngrams(text: Union[str, List[str]], n: int) -> List[List[str]]:
    """
    Generate n-grams from a string or list of tokens.
    
    Args:
        text: The string or list of tokens to generate n-grams from
        n: The size of the n-grams
        
    Returns:
        The list of n-grams
    """
    if isinstance(text, str):
        tokens = tokenize(text)
    else:
        tokens = text
    
    return [tokens[i:i+n] for i in range(len(tokens) - n + 1)]