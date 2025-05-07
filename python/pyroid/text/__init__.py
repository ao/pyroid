"""
Pyroid Text Module
================

This module provides high-performance text processing operations.

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

# Import from our Python implementation
from ..text_impl import (
    # String operations
    reverse,
    base64_encode,
    base64_decode,
    split,
    join,
    replace,
    regex_replace,
    to_uppercase,
    to_lowercase,
    
    # NLP operations
    tokenize,
    ngrams,
)

__all__ = [
    'reverse',
    'base64_encode',
    'base64_decode',
    'split',
    'join',
    'replace',
    'regex_replace',
    'tokenize',
    'ngrams',
    'to_uppercase',
    'to_lowercase',
]