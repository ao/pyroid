#!/usr/bin/env python3
"""
Text and NLP operation examples for pyroid.

This script demonstrates the text and NLP capabilities of pyroid.
"""

import time
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def main():
    print("Pyroid Text and NLP Operations Examples")
    print("===================================")
    
    # Example 1: Basic text operations
    print("\n1. Basic Text Operations")
    
    # Create a test string
    text = "Hello, World! This is a test of Pyroid's text processing capabilities."
    print(f"Original text: {text}")
    
    # Reverse the string
    reversed_text = pyroid.text.reverse(text)
    print(f"Reversed: {reversed_text}")
    
    # Convert to uppercase
    upper = pyroid.text.to_uppercase(text)
    print(f"Uppercase: {upper}")
    
    # Convert to lowercase
    lower = pyroid.text.to_lowercase(text)
    print(f"Lowercase: {lower}")
    
    # Example 2: String manipulation
    print("\n2. String Manipulation")
    
    # Split the string
    words = pyroid.text.split(text, " ")
    print(f"Split by space: {words}")
    
    # Join the words
    joined = pyroid.text.join(words, "-")
    print(f"Joined with hyphens: {joined}")
    
    # Regex replace
    replaced = pyroid.text.regex_replace(text, r"World", "Python")
    print(f"Regex replace 'World' with 'Python': {replaced}")
    
    # Example 3: Base64 encoding/decoding
    print("\n3. Base64 Encoding/Decoding")
    
    # Encode to base64
    encoded = pyroid.text.base64_encode(text)
    print(f"Base64 encoded: {encoded}")
    
    # Decode from base64
    decoded = pyroid.text.base64_decode(encoded)
    print(f"Base64 decoded: {decoded}")
    print(f"Original and decoded match: {text == decoded}")
    
    # Example 4: NLP operations
    print("\n4. NLP Operations")
    
    # Tokenize the text
    tokens = pyroid.text.tokenize(text)
    print(f"Tokenized: {tokens}")
    
    # Generate n-grams
    ngrams = pyroid.text.ngrams(text, 2)
    print(f"Bigrams: {ngrams}")
    
    # Try stemming (may not be implemented)
    try:
        stemmed = pyroid.text.stem(text)
        print(f"Stemmed: {stemmed}")
    except Exception as e:
        print(f"Stemming not implemented: {e}")
    
    # Try lemmatization (may not be implemented)
    try:
        lemmatized = pyroid.text.lemmatize(text)
        print(f"Lemmatized: {lemmatized}")
    except Exception as e:
        print(f"Lemmatization not implemented: {e}")
    
    # Try part-of-speech tagging (may not be implemented)
    try:
        pos_tags = pyroid.text.pos_tag(text)
        print(f"POS Tags: {pos_tags}")
    except Exception as e:
        print(f"POS tagging not implemented: {e}")
    
    # Example 5: Text analysis
    print("\n5. Text Analysis")
    
    # Try sentiment analysis (may not be implemented)
    try:
        positive_text = "I love this product! It's amazing and works perfectly."
        negative_text = "This is terrible. I hate it and it doesn't work at all."
        
        positive_sentiment = pyroid.text.sentiment(positive_text)
        negative_sentiment = pyroid.text.sentiment(negative_text)
        
        print(f"Positive text sentiment: {positive_sentiment}")
        print(f"Negative text sentiment: {negative_sentiment}")
    except Exception as e:
        print(f"Sentiment analysis not implemented: {e}")
    
    # Try keyword extraction (may not be implemented)
    try:
        article = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
        AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
        The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving".
        This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
        """
        
        keywords = pyroid.text.extract_keywords(article, n=5)
        print(f"Keywords: {keywords}")
    except Exception as e:
        print(f"Keyword extraction not implemented: {e}")
    
    # Try text similarity (may not be implemented)
    try:
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "A fast brown fox leaps over a lazy dog."
        
        similarity = pyroid.text.similarity(text1, text2)
        print(f"Text similarity: {similarity}")
    except Exception as e:
        print(f"Text similarity not implemented: {e}")
    
    # Example 6: Performance comparison
    print("\n6. Performance Comparison")
    
    # Create a large text for benchmarking
    large_text = "Hello, World! " * 10000
    print(f"Large text length: {len(large_text)}")
    
    # Benchmark Python's built-in uppercase
    print("\nPython built-in uppercase:")
    python_result = benchmark(lambda: large_text.upper())
    print(f"Result length: {len(python_result)}")
    
    # Benchmark Pyroid's uppercase
    print("\nPyroid uppercase:")
    pyroid_result = benchmark(lambda: pyroid.text.to_uppercase(large_text))
    print(f"Result length: {len(pyroid_result)}")
    
    # Benchmark Python's built-in split
    print("\nPython built-in split:")
    python_result = benchmark(lambda: large_text.split(" "))
    print(f"Result length: {len(python_result)}")
    
    # Benchmark Pyroid's split
    print("\nPyroid split:")
    pyroid_result = benchmark(lambda: pyroid.text.split(large_text, " "))
    print(f"Result length: {len(pyroid_result)}")

if __name__ == "__main__":
    main()