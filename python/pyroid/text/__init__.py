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
    tokenize: Tokenize a string into words
    ngrams: Generate n-grams from a string
"""

# Try to import directly from the pyroid module
try:
    from ..pyroid import (
        # String operations
        reverse,
        base64_encode,
        base64_decode,
        split,
        join,
        replace,
        regex_replace,
        
        # NLP operations
        tokenize,
        ngrams,
    )
except ImportError:
    # Fallback to importing from the text module
    try:
        from ..pyroid.text import (
            # String operations
            reverse,
            base64_encode,
            base64_decode,
            split,
            join,
            replace,
            regex_replace,
            
            # NLP operations
            tokenize,
            ngrams,
        )
    except ImportError:
        # If all else fails, create dummy functions for documentation purposes
        def reverse(text):
            """Reverse a string."""
            return text[::-1]
            
        def base64_encode(text):
            """Encode a string to base64."""
            import base64
            return base64.b64encode(text.encode()).decode()
            
        def base64_decode(text):
            """Decode a base64 string."""
            import base64
            return base64.b64decode(text.encode()).decode()
            
        def split(text, delimiter):
            """Split a string by a delimiter."""
            return text.split(delimiter)
            
        def join(strings, delimiter):
            """Join a list of strings with a delimiter."""
            return delimiter.join(strings)
            
        def replace(text, old, new):
            """Replace a substring in a string."""
            return text.replace(old, new)
            
        def regex_replace(text, pattern, replacement):
            """Replace a regex pattern in a string."""
            import re
            return re.sub(pattern, replacement, text)
            
        def tokenize(text, lowercase=True, remove_punct=True):
            """Tokenize a string into words."""
            import re
            if lowercase:
                text = text.lower()
            if remove_punct:
                text = re.sub(r'[^\w\s]', '', text)
            return text.split()
            
        def ngrams(text, n, lowercase=True):
            """Generate n-grams from a string."""
            if lowercase:
                text = text.lower()
            tokens = text.split()
            return [tokens[i:i+n] for i in range(len(tokens) - n + 1)]

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
]