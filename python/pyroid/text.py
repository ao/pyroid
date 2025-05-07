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

# Import directly from Rust extension
try:
    from .pyroid import (
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
except ImportError as e:
    error_message = f"""
    ERROR: Pyroid text operations could not be loaded!
    
    Pyroid requires the text Rust extensions to be properly built and installed.
    
    Error: {str(e)}
    
    To fix this:
    1. Make sure you've installed pyroid with the Rust components:
       python build_and_install.py
    2. Check that the Rust toolchain is properly installed
    3. Verify that the compiled extensions (.so/.pyd files) exist in the package directory
    
    For more help, visit: https://github.com/ao/pyroid/issues
    """
    raise ImportError(error_message)

__all__ = [
    # String operations
    'reverse',
    'base64_encode',
    'base64_decode',
    'split',
    'join',
    'replace',
    'regex_replace',
    'to_uppercase',
    'to_lowercase',
    
    # NLP operations
    'tokenize',
    'ngrams',
]