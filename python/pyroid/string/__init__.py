"""
Pyroid String Module
=================

This module provides high-performance string operations.
It is an alias for the text module for backward compatibility.

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
"""

# Import all functions from the text module
from ..text import (
    reverse,
    base64_encode,
    base64_decode,
    split,
    join,
    replace,
    regex_replace,
    tokenize,
    ngrams,
    to_uppercase,
    to_lowercase,
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