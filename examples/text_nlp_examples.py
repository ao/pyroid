#!/usr/bin/env python3
"""
Text and NLP operation examples for pyroid.

This script demonstrates the text and NLP capabilities of pyroid.
"""

import time
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def generate_random_text(word_count, word_length=5):
    """Generate random text with the specified number of words."""
    words = []
    for _ in range(word_count):
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
    return ' '.join(words)

def main():
    print("pyroid Text and NLP Operations Examples")
    print("=====================================")
    
    # Example 1: Tokenization
    print("\n1. Tokenization")
    
    # Generate sample texts
    n_texts = 10000
    texts = [generate_random_text(50) for _ in range(n_texts)]
    
    print(f"\nTokenizing {n_texts} texts:")
    
    print("\nNLTK tokenization:")
    nltk_result = benchmark(lambda: [word_tokenize(text) for text in texts])
    
    print("\npyroid parallel tokenization:")
    pyroid_result = benchmark(lambda: pyroid.parallel_tokenize(texts, True, True))
    
    print("\nResults (first text):")
    print(f"NLTK: {nltk_result[0][:10]}...")
    print(f"pyroid: {pyroid_result[0][:10]}...")
    
    # Example 2: N-grams
    print("\n2. N-grams")
    
    # Generate sample texts
    n_texts = 5000
    texts = [generate_random_text(100) for _ in range(n_texts)]
    
    print(f"\nGenerating bigrams for {n_texts} texts:")
    
    print("\nPython n-grams:")
    def python_ngrams(texts, n):
        results = []
        for text in texts:
            tokens = text.split()
            if len(tokens) >= n:
                ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                results.append(ngrams)
            else:
                results.append([])
        return results
    
    python_result = benchmark(lambda: python_ngrams(texts, 2))
    
    print("\npyroid parallel n-grams:")
    pyroid_result = benchmark(lambda: pyroid.parallel_ngrams(texts, 2, False))
    
    print("\nResults (first text):")
    print(f"Python: {python_result[0][:5]}...")
    print(f"pyroid: {pyroid_result[0][:5]}...")
    
    # Example 3: TF-IDF
    print("\n3. TF-IDF")
    
    # Generate sample documents
    n_docs = 1000
    docs = [generate_random_text(100) for _ in range(n_docs)]
    
    print(f"\nCalculating TF-IDF for {n_docs} documents:")
    
    print("\nScikit-learn TfidfVectorizer:")
    vectorizer = TfidfVectorizer()
    sklearn_result = benchmark(lambda: vectorizer.fit_transform(docs))
    
    print("\npyroid parallel TF-IDF:")
    pyroid_result = benchmark(lambda: pyroid.parallel_tfidf(docs, False))
    
    print("\nResults (shape):")
    print(f"Scikit-learn: {sklearn_result.shape}")
    print(f"pyroid: (tfidf_matrix: {len(pyroid_result[0])}, vocabulary: {len(pyroid_result[1])})")
    
    # Example 4: Document Similarity
    print("\n4. Document Similarity")
    
    # Generate sample documents
    n_docs = 500
    docs = [generate_random_text(50) for _ in range(n_docs)]
    
    print(f"\nCalculating document similarity for {n_docs} documents:")
    
    print("\nScikit-learn cosine similarity:")
    def sklearn_doc_similarity(docs):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs)
        return cosine_similarity(tfidf_matrix)
    
    sklearn_result = benchmark(lambda: sklearn_doc_similarity(docs))
    
    print("\npyroid parallel document similarity:")
    pyroid_result = benchmark(lambda: pyroid.parallel_document_similarity(docs, "cosine", False))
    
    print("\nResults (shape):")
    print(f"Scikit-learn: {sklearn_result.shape}")
    print(f"pyroid: ({len(pyroid_result)}, {len(pyroid_result[0])})")

if __name__ == "__main__":
    main()