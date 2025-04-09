"""
Text processing pipeline benchmark for Pyroid.

This module provides a benchmark that simulates a text processing pipeline
to showcase Pyroid's performance advantages in NLP preprocessing tasks.
"""

import re
import time
import random
import string

try:
    import pyroid
except ImportError:
    print("Warning: pyroid not found. Text processing benchmark will not run correctly.")

from ...core.benchmark import Benchmark
from ...core.reporter import BenchmarkReporter


def generate_text_corpus(size=10_000, avg_length=100):
    """Generate a random text corpus for benchmarking.
    
    Args:
        size: Number of documents to generate.
        avg_length: Average length of each document in words.
        
    Returns:
        A list of text documents.
    """
    print(f"Generating {size:,} documents with average length of {avg_length} words...")
    
    # Common English words for more realistic text
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"
    ]
    
    # Punctuation to add
    punctuation = ".,.!?;:"
    
    # Generate documents
    documents = []
    for i in range(size):
        # Vary the length around the average
        length = max(10, int(random.gauss(avg_length, avg_length / 4)))
        
        # Generate words
        words = []
        for j in range(length):
            # 90% common words, 10% random strings
            if random.random() < 0.9:
                word = random.choice(common_words)
            else:
                word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            
            # Randomly capitalize some words
            if random.random() < 0.1:
                word = word.capitalize()
            
            # Randomly add punctuation
            if random.random() < 0.1:
                word += random.choice(punctuation)
            
            words.append(word)
        
        # Join words into a document
        document = ' '.join(words)
        
        # Add some special characters and numbers
        if random.random() < 0.3:
            document += f" #{random.randint(1, 1000)}"
        
        if random.random() < 0.2:
            document += f" @user{random.randint(1, 100)}"
        
        documents.append(document)
    
    print("Text corpus generation complete.")
    return documents


def run_text_processing_benchmark(size=10_000):
    """Run a text processing pipeline benchmark.
    
    Args:
        size: Number of documents to process.
        
    Returns:
        A Benchmark object with results.
    """
    # Generate test data
    documents = generate_text_corpus(size)
    
    pipeline_benchmark = Benchmark("Text Processing Pipeline", f"NLP preprocessing on {size:,} documents")
    
    # Common English stopwords
    stopwords = set([
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what"
    ])
    
    # Python implementation
    def python_pipeline(documents):
        print("Running Python text processing pipeline...")
        
        # Step 1: Clean text (remove special characters, normalize case)
        print("  Step 1: Cleaning text...")
        cleaned = []
        for doc in documents:
            # Convert to lowercase
            doc = doc.lower()
            # Remove special characters
            doc = re.sub(r'[^a-z0-9\s]', '', doc)
            # Normalize whitespace
            doc = re.sub(r'\s+', ' ', doc).strip()
            cleaned.append(doc)
        
        # Step 2: Tokenize
        print("  Step 2: Tokenizing...")
        tokenized = []
        for doc in cleaned:
            tokens = doc.split()
            tokenized.append(tokens)
        
        # Step 3: Remove stopwords
        print("  Step 3: Removing stopwords...")
        filtered = []
        for tokens in tokenized:
            filtered_tokens = [token for token in tokens if token not in stopwords]
            filtered.append(filtered_tokens)
        
        # Step 4: Calculate term frequencies
        print("  Step 4: Calculating term frequencies...")
        term_freqs = []
        for tokens in filtered:
            freq = {}
            for token in tokens:
                if token in freq:
                    freq[token] += 1
                else:
                    freq[token] = 1
            term_freqs.append(freq)
        
        # Step 5: Find top terms
        print("  Step 5: Finding top terms...")
        top_terms = {}
        for freq in term_freqs:
            for term, count in freq.items():
                if term in top_terms:
                    top_terms[term] += count
                else:
                    top_terms[term] = count
        
        # Sort by frequency
        sorted_terms = sorted(top_terms.items(), key=lambda x: x[1], reverse=True)
        top_100 = sorted_terms[:100]
        
        print("Python text processing pipeline complete.")
        return {
            "document_count": len(documents),
            "cleaned_count": len(cleaned),
            "tokenized_count": len(tokenized),
            "filtered_count": len(filtered),
            "term_freqs_count": len(term_freqs),
            "top_terms": top_100
        }
    
    # pyroid implementation
    def pyroid_pipeline(documents):
        print("Running pyroid text processing pipeline...")
        
        # Step 1: Clean text (remove special characters, normalize case)
        print("  Step 1: Cleaning text...")
        
        def clean_text(doc):
            # Convert to lowercase
            doc = doc.lower()
            # Remove special characters
            doc = re.sub(r'[^a-z0-9\s]', '', doc)
            # Normalize whitespace
            doc = re.sub(r'\s+', ' ', doc).strip()
            return doc
        
        # Use data.collections.map or fallback to map
        try:
            cleaned = pyroid.data.collections.map(documents, clean_text)
        except AttributeError:
            cleaned = list(map(clean_text, documents))
        
        # Step 2: Tokenize
        print("  Step 2: Tokenizing...")
        # Use data.collections.map or fallback to map
        try:
            tokenized = pyroid.data.collections.map(cleaned, lambda doc: doc.split())
        except AttributeError:
            tokenized = list(map(lambda doc: doc.split(), cleaned))
        
        # Step 3: Remove stopwords
        print("  Step 3: Removing stopwords...")
        
        def remove_stopwords(tokens):
            return [token for token in tokens if token not in stopwords]
        
        # Use data.collections.map or fallback to map
        try:
            filtered = pyroid.data.collections.map(tokenized, remove_stopwords)
        except AttributeError:
            filtered = list(map(remove_stopwords, tokenized))
        
        # Step 4: Calculate term frequencies
        print("  Step 4: Calculating term frequencies...")
        
        def calculate_freq(tokens):
            freq = {}
            for token in tokens:
                if token in freq:
                    freq[token] += 1
                else:
                    freq[token] = 1
            return freq
        
        # Use data.collections.map or fallback to map
        try:
            term_freqs = pyroid.data.collections.map(filtered, calculate_freq)
        except AttributeError:
            term_freqs = list(map(calculate_freq, filtered))
        
        # Step 5: Find top terms
        print("  Step 5: Finding top terms...")
        top_terms = {}
        for freq in term_freqs:
            for term, count in freq.items():
                if term in top_terms:
                    top_terms[term] += count
                else:
                    top_terms[term] = count
        
        # Sort by frequency
        sorted_terms = sorted(top_terms.items(), key=lambda x: x[1], reverse=True)
        top_100 = sorted_terms[:100]
        
        print("pyroid text processing pipeline complete.")
        return {
            "document_count": len(documents),
            "cleaned_count": len(cleaned),
            "tokenized_count": len(tokenized),
            "filtered_count": len(filtered),
            "term_freqs_count": len(term_freqs),
            "top_terms": top_100
        }
    
    # Set appropriate timeouts
    python_timeout = 30  # Complex pipeline might take longer
    pyroid_timeout = 10
    
    pipeline_benchmark.run_test("Python text pipeline", "Python", python_pipeline, python_timeout, documents)
    pipeline_benchmark.run_test("pyroid text pipeline", "pyroid", pyroid_pipeline, pyroid_timeout, documents)
    
    BenchmarkReporter.print_results(pipeline_benchmark)
    return pipeline_benchmark


if __name__ == "__main__":
    print("Running text processing pipeline benchmark...")
    run_text_processing_benchmark()