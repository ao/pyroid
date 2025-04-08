# Text and NLP Operations

This document provides detailed information about the text and natural language processing operations available in Pyroid.

## Basic Text Operations

### Reverse

```python
pyroid.text.reverse(text)
```

Reverses the characters in a string.

**Parameters:**
- `text` (str): The input string to reverse.

**Returns:**
- The reversed string.

**Example:**
```python
import pyroid

text = "Hello, World!"
reversed_text = pyroid.text.reverse(text)
print(f"Original: {text}")
print(f"Reversed: {reversed_text}")
```

### To Uppercase

```python
pyroid.text.to_uppercase(text)
```

Converts a string to uppercase.

**Parameters:**
- `text` (str): The input string to convert.

**Returns:**
- The uppercase string.

**Example:**
```python
import pyroid

text = "Hello, World!"
uppercase = pyroid.text.to_uppercase(text)
print(f"Original: {text}")
print(f"Uppercase: {uppercase}")
```

### To Lowercase

```python
pyroid.text.to_lowercase(text)
```

Converts a string to lowercase.

**Parameters:**
- `text` (str): The input string to convert.

**Returns:**
- The lowercase string.

**Example:**
```python
import pyroid

text = "Hello, World!"
lowercase = pyroid.text.to_lowercase(text)
print(f"Original: {text}")
print(f"Lowercase: {lowercase}")
```

## String Manipulation

### Split

```python
pyroid.text.split(text, delimiter)
```

Splits a string by a delimiter.

**Parameters:**
- `text` (str): The input string to split.
- `delimiter` (str): The delimiter to split by.

**Returns:**
- A list of substrings.

**Example:**
```python
import pyroid

text = "Hello, World!"
words = pyroid.text.split(text, " ")
print(f"Original: {text}")
print(f"Split: {words}")
```

### Join

```python
pyroid.text.join(parts, delimiter)
```

Joins a list of strings with a delimiter.

**Parameters:**
- `parts` (list): The list of strings to join.
- `delimiter` (str): The delimiter to join with.

**Returns:**
- The joined string.

**Example:**
```python
import pyroid

words = ["Hello", "World"]
joined = pyroid.text.join(words, "-")
print(f"Parts: {words}")
print(f"Joined: {joined}")
```

### Regex Replace

```python
pyroid.text.regex_replace(text, pattern, replacement)
```

Replaces occurrences of a regex pattern in a string.

**Parameters:**
- `text` (str): The input string.
- `pattern` (str): The regex pattern to search for.
- `replacement` (str): The replacement string.

**Returns:**
- The string with replacements.

**Example:**
```python
import pyroid

text = "Hello, World!"
replaced = pyroid.text.regex_replace(text, r"World", "Python")
print(f"Original: {text}")
print(f"Replaced: {replaced}")
```

## Encoding/Decoding

### Base64 Encode

```python
pyroid.text.base64_encode(text)
```

Encodes a string to base64.

**Parameters:**
- `text` (str): The input string to encode.

**Returns:**
- The base64-encoded string.

**Example:**
```python
import pyroid

text = "Hello, World!"
encoded = pyroid.text.base64_encode(text)
print(f"Original: {text}")
print(f"Encoded: {encoded}")
```

### Base64 Decode

```python
pyroid.text.base64_decode(encoded)
```

Decodes a base64-encoded string.

**Parameters:**
- `encoded` (str): The base64-encoded string to decode.

**Returns:**
- The decoded string.

**Example:**
```python
import pyroid

encoded = "SGVsbG8sIFdvcmxkIQ=="
decoded = pyroid.text.base64_decode(encoded)
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

## NLP Operations

### Tokenize

```python
pyroid.text.tokenize(text)
```

Tokenizes a string into words.

**Parameters:**
- `text` (str): The input string to tokenize.

**Returns:**
- A list of tokens.

**Example:**
```python
import pyroid

text = "Hello, World!"
tokens = pyroid.text.tokenize(text)
print(f"Original: {text}")
print(f"Tokens: {tokens}")
```

### N-grams

```python
pyroid.text.ngrams(text, n)
```

Generates n-grams from a string.

**Parameters:**
- `text` (str): The input string.
- `n` (int): The size of the n-grams.

**Returns:**
- A list of n-grams.

**Example:**
```python
import pyroid

text = "Hello, World!"
bigrams = pyroid.text.ngrams(text, 2)
print(f"Original: {text}")
print(f"Bigrams: {bigrams}")
```

### Stemming

```python
pyroid.text.stem(text)
```

Stems words in a string.

**Parameters:**
- `text` (str): The input string.

**Returns:**
- The string with stemmed words.

**Example:**
```python
import pyroid

text = "running jumps easily"
stemmed = pyroid.text.stem(text)
print(f"Original: {text}")
print(f"Stemmed: {stemmed}")
```

### Lemmatization

```python
pyroid.text.lemmatize(text)
```

Lemmatizes words in a string.

**Parameters:**
- `text` (str): The input string.

**Returns:**
- The string with lemmatized words.

**Example:**
```python
import pyroid

text = "running jumps easily"
lemmatized = pyroid.text.lemmatize(text)
print(f"Original: {text}")
print(f"Lemmatized: {lemmatized}")
```

### Part-of-Speech Tagging

```python
pyroid.text.pos_tag(text)
```

Tags parts of speech in a string.

**Parameters:**
- `text` (str): The input string.

**Returns:**
- A list of (token, tag) tuples.

**Example:**
```python
import pyroid

text = "The quick brown fox jumps over the lazy dog."
pos_tags = pyroid.text.pos_tag(text)
print(f"Original: {text}")
print(f"POS Tags: {pos_tags}")
```

### Named Entity Recognition

```python
pyroid.text.ner(text)
```

Recognizes named entities in a string.

**Parameters:**
- `text` (str): The input string.

**Returns:**
- A list of (entity, type) tuples.

**Example:**
```python
import pyroid

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
entities = pyroid.text.ner(text)
print(f"Original: {text}")
print(f"Named Entities: {entities}")
```

### Sentiment Analysis

```python
pyroid.text.sentiment(text)
```

Analyzes the sentiment of a string.

**Parameters:**
- `text` (str): The input string.

**Returns:**
- A dictionary containing sentiment scores.

**Example:**
```python
import pyroid

text = "I love this product! It's amazing."
sentiment = pyroid.text.sentiment(text)
print(f"Original: {text}")
print(f"Sentiment: {sentiment}")
```

### Text Summarization

```python
pyroid.text.summarize(text, ratio=0.2)
```

Summarizes a text.

**Parameters:**
- `text` (str): The input text to summarize.
- `ratio` (float, optional): The ratio of the original text to keep. Default is 0.2.

**Returns:**
- The summarized text.

**Example:**
```python
import pyroid

text = """
Pyroid is a high-performance Rust-powered library for Python that accelerates common operations and eliminates performance bottlenecks.
It provides optimized implementations of various operations across multiple domains, including math, string processing, data manipulation, I/O, image processing, and machine learning.
The library is designed to be easy to use with a Pythonic API, while leveraging the performance benefits of Rust.
"""

summary = pyroid.text.summarize(text, ratio=0.5)
print(f"Original length: {len(text)}")
print(f"Summary length: {len(summary)}")
print(f"Summary: {summary}")
```

### Keyword Extraction

```python
pyroid.text.extract_keywords(text, n=5)
```

Extracts keywords from a text.

**Parameters:**
- `text` (str): The input text.
- `n` (int, optional): The number of keywords to extract. Default is 5.

**Returns:**
- A list of keywords.

**Example:**
```python
import pyroid

text = """
Pyroid is a high-performance Rust-powered library for Python that accelerates common operations and eliminates performance bottlenecks.
It provides optimized implementations of various operations across multiple domains, including math, string processing, data manipulation, I/O, image processing, and machine learning.
The library is designed to be easy to use with a Pythonic API, while leveraging the performance benefits of Rust.
"""

keywords = pyroid.text.extract_keywords(text, n=5)
print(f"Keywords: {keywords}")
```

### Text Classification

```python
pyroid.text.classify(text, categories)
```

Classifies a text into one of the provided categories.

**Parameters:**
- `text` (str): The input text to classify.
- `categories` (list): A list of possible categories.

**Returns:**
- The most likely category and a confidence score.

**Example:**
```python
import pyroid

text = "The stock market experienced significant growth today."
categories = ["business", "sports", "technology", "politics"]
classification = pyroid.text.classify(text, categories)
print(f"Text: {text}")
print(f"Classification: {classification}")
```

### Language Detection

```python
pyroid.text.detect_language(text)
```

Detects the language of a text.

**Parameters:**
- `text` (str): The input text.

**Returns:**
- The detected language code and a confidence score.

**Example:**
```python
import pyroid

text = "Hello, World!"
language = pyroid.text.detect_language(text)
print(f"Text: {text}")
print(f"Detected Language: {language}")

text_fr = "Bonjour le monde!"
language_fr = pyroid.text.detect_language(text_fr)
print(f"Text: {text_fr}")
print(f"Detected Language: {language_fr}")
```

### Text Similarity

```python
pyroid.text.similarity(text1, text2, method="cosine")
```

Calculates the similarity between two texts.

**Parameters:**
- `text1` (str): The first text.
- `text2` (str): The second text.
- `method` (str, optional): The similarity method. One of "cosine", "jaccard", "levenshtein". Default is "cosine".

**Returns:**
- A similarity score between 0 and 1.

**Example:**
```python
import pyroid

text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A fast brown fox leaps over a lazy dog."
similarity = pyroid.text.similarity(text1, text2, method="cosine")
print(f"Text 1: {text1}")
print(f"Text 2: {text2}")
print(f"Similarity: {similarity}")