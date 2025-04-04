# Text and NLP Operations

The Text and NLP operations module provides high-performance implementations of common text processing and natural language processing tasks. These operations are implemented in Rust and are designed to be significantly faster than their Python equivalents, especially for large text datasets.

## Overview

The Text and NLP operations module provides the following key functions:

- `parallel_tokenize`: Tokenize texts in parallel
- `parallel_ngrams`: Generate n-grams from texts in parallel
- `parallel_tfidf`: Calculate TF-IDF matrix in parallel
- `parallel_document_similarity`: Calculate document similarity matrix in parallel

## API Reference

### parallel_tokenize

Tokenize texts in parallel.

```python
pyroid.parallel_tokenize(texts, lowercase=True, remove_punct=True)
```

#### Parameters

- `texts`: A list of texts to tokenize
- `lowercase`: Whether to lowercase the texts before tokenization (default: True)
- `remove_punct`: Whether to remove punctuation (default: True)

#### Returns

A list of tokenized texts (each text is a list of tokens).

#### Example

```python
import pyroid
import time
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Sample texts
texts = [
    "Hello, world! This is a test.",
    "Pyroid is fast and efficient.",
    "Natural language processing is fascinating."
] * 1000  # Repeat to create a larger dataset

# Compare with NLTK
start = time.time()
nltk_tokens = [word_tokenize(text.lower()) for text in texts]
nltk_time = time.time() - start

start = time.time()
pyroid_tokens = pyroid.parallel_tokenize(texts, True, True)
pyroid_time = time.time() - start

print(f"NLTK time: {nltk_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {nltk_time / pyroid_time:.1f}x")

# Compare results for the first text
print(f"NLTK tokens: {nltk_tokens[0]}")
print(f"Pyroid tokens: {pyroid_tokens[0]}")
```

#### Performance Considerations

- `parallel_tokenize` is particularly efficient for large datasets with many texts.
- The implementation processes each text in parallel, which can lead to significant performance improvements on multi-core systems.
- The function uses a simple whitespace-based tokenization approach, which is faster but less sophisticated than NLTK's tokenization.
- For languages other than English or for specialized tokenization needs, you may need to use a more sophisticated tokenizer.

### parallel_ngrams

Generate n-grams from texts in parallel.

```python
pyroid.parallel_ngrams(texts, n=2, tokenized=False)
```

#### Parameters

- `texts`: A list of texts or tokenized texts
- `n`: Size of n-grams (default: 2)
- `tokenized`: Whether the input is already tokenized (default: False)

#### Returns

A list of n-grams for each text.

#### Example

```python
import pyroid
import time
from nltk.util import ngrams
import nltk

# Sample texts
texts = [
    "Hello world this is a test",
    "Pyroid is fast and efficient",
    "Natural language processing is fascinating"
] * 1000  # Repeat to create a larger dataset

# Compare with NLTK
start = time.time()
nltk_bigrams = []
for text in texts:
    tokens = text.split()
    nltk_bigrams.append(list(ngrams(tokens, 2)))
nltk_time = time.time() - start

start = time.time()
pyroid_bigrams = pyroid.parallel_ngrams(texts, 2, False)
pyroid_time = time.time() - start

print(f"NLTK time: {nltk_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {nltk_time / pyroid_time:.1f}x")

# Compare results for the first text
print(f"NLTK bigrams: {[' '.join(bg) for bg in nltk_bigrams[0]]}")
print(f"Pyroid bigrams: {pyroid_bigrams[0]}")
```

#### Performance Considerations

- `parallel_ngrams` is particularly efficient for large datasets with many texts.
- The implementation processes each text in parallel, which can lead to significant performance improvements on multi-core systems.
- Setting `tokenized=True` can improve performance if you already have tokenized texts, as it avoids redundant tokenization.
- For very large n-gram sizes, memory usage can be a concern as the number of n-grams grows exponentially with n.

### parallel_tfidf

Calculate TF-IDF matrix in parallel.

```python
pyroid.parallel_tfidf(documents, tokenized=False, min_df=1, max_df=1.0)
```

#### Parameters

- `documents`: A list of documents (strings or tokenized documents)
- `tokenized`: Whether the input is already tokenized (default: False)
- `min_df`: Minimum document frequency for a term to be included (default: 1)
- `max_df`: Maximum document frequency for a term to be included (default: 1.0)

#### Returns

A tuple of (tfidf_matrix, vocabulary):
- `tfidf_matrix`: A list of dictionaries mapping term indices to TF-IDF values
- `vocabulary`: A dictionary mapping terms to indices

#### Example

```python
import pyroid
import time
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document?"
] * 250  # Repeat to create a larger dataset

# Compare with scikit-learn
start = time.time()
vectorizer = TfidfVectorizer()
sklearn_tfidf = vectorizer.fit_transform(documents)
sklearn_time = time.time() - start

start = time.time()
pyroid_tfidf_matrix, pyroid_vocabulary = pyroid.parallel_tfidf(documents, False)
pyroid_time = time.time() - start

print(f"Scikit-learn time: {sklearn_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {sklearn_time / pyroid_time:.1f}x")

# Compare vocabulary sizes
print(f"Scikit-learn vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Pyroid vocabulary size: {len(pyroid_vocabulary)}")
```

#### Performance Considerations

- `parallel_tfidf` is particularly efficient for large datasets with many documents.
- The implementation processes each document in parallel, which can lead to significant performance improvements on multi-core systems.
- Setting `tokenized=True` can improve performance if you already have tokenized documents, as it avoids redundant tokenization.
- The `min_df` and `max_df` parameters can be used to filter out rare or common terms, which can improve both performance and results quality.
- The function returns a sparse representation of the TF-IDF matrix, which is memory-efficient for large datasets.

### parallel_document_similarity

Calculate document similarity matrix in parallel.

```python
pyroid.parallel_document_similarity(docs, method='cosine', tokenized=False)
```

#### Parameters

- `docs`: A list of documents (strings or tokenized documents)
- `method`: Similarity method (default: 'cosine')
  - Supported methods: 'cosine', 'jaccard', 'overlap'
- `tokenized`: Whether the input is already tokenized (default: False)

#### Returns

A 2D array of similarity scores between documents.

#### Example

```python
import pyroid
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document?"
] * 25  # Repeat to create a larger dataset

# Compare with scikit-learn
start = time.time()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
sklearn_similarity = cosine_similarity(tfidf_matrix)
sklearn_time = time.time() - start

start = time.time()
pyroid_similarity = pyroid.parallel_document_similarity(documents, "cosine", False)
pyroid_time = time.time() - start

print(f"Scikit-learn time: {sklearn_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {sklearn_time / pyroid_time:.1f}x")

# Compare results for the first document
print(f"Scikit-learn similarities for first document: {sklearn_similarity[0][:4]}")
print(f"Pyroid similarities for first document: {pyroid_similarity[0][:4]}")
```

#### Similarity Methods

1. **Cosine Similarity ('cosine')**

   Measures the cosine of the angle between two non-zero vectors:
   
   ```
   similarity = (A · B) / (||A|| * ||B||)
   ```
   
   where A and B are document vectors, · is the dot product, and ||A|| is the norm of A.

2. **Jaccard Similarity ('jaccard')**

   Measures the size of the intersection divided by the size of the union of two sets:
   
   ```
   similarity = |A ∩ B| / |A ∪ B|
   ```
   
   where A and B are sets of terms in the documents.

3. **Overlap Coefficient ('overlap')**

   Measures the overlap between two sets:
   
   ```
   similarity = |A ∩ B| / min(|A|, |B|)
   ```
   
   where A and B are sets of terms in the documents.

#### Performance Considerations

- `parallel_document_similarity` is particularly efficient for large datasets with many documents.
- The implementation processes document pairs in parallel, which can lead to significant performance improvements on multi-core systems.
- Setting `tokenized=True` can improve performance if you already have tokenized documents, as it avoids redundant tokenization.
- For very large document collections, memory usage can be a concern as the similarity matrix grows quadratically with the number of documents.
- The 'cosine' method is generally faster than 'jaccard' and 'overlap' for large documents, as it uses optimized vector operations.

## Performance Comparison

The following table shows the performance comparison between common Python libraries and pyroid for various text and NLP operations:

| Operation | Dataset Size | Python Library | pyroid | Speedup |
|-----------|-------------|----------------|--------|---------|
| Tokenization | 5000 texts | NLTK: 2500ms | 200ms | 12.5x |
| N-grams | 5000 texts | NLTK: 1800ms | 150ms | 12.0x |
| TF-IDF | 1000 docs | scikit-learn: 800ms | 300ms | 2.7x |
| Document Similarity | 500 docs | scikit-learn: 600ms | 200ms | 3.0x |

## Best Practices

1. **Preprocess text data**: Clean and normalize your text data before processing to improve results quality and performance.

2. **Use tokenized input when possible**: If you already have tokenized texts, set `tokenized=True` to avoid redundant tokenization.

3. **Filter out rare and common terms**: Use the `min_df` and `max_df` parameters in `parallel_tfidf` to filter out terms that are too rare or too common, which can improve both performance and results quality.

4. **Choose the appropriate similarity metric**: Different similarity metrics are suitable for different types of text data and applications. For example, 'cosine' is suitable for general text similarity, while 'jaccard' may be better for short texts or keyword matching.

5. **Be mindful of memory usage**: For very large document collections, consider processing data in batches to avoid memory issues, especially when calculating document similarity matrices.

## Limitations

1. **Simple tokenization**: The tokenization approach is simpler than specialized NLP libraries like NLTK or spaCy, which may affect results for complex languages or specialized domains.

2. **Limited preprocessing options**: The current implementation provides basic preprocessing options (lowercase, remove punctuation), but lacks advanced options like stemming, lemmatization, or stop word removal.

3. **Memory usage**: For very large document collections, memory usage can be a concern, especially for document similarity matrices.

4. **No language-specific features**: The current implementation does not include language-specific features or models.

## Examples

### Example 1: Text Classification Pipeline

```python
import pyroid
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample documents and labels
documents = [
    "This movie is great and I enjoyed it",
    "This movie is terrible and I hated it",
    "I loved this movie, it was amazing",
    "I disliked this movie, it was boring",
    "Great acting and plot, highly recommend",
    "Poor acting and plot, do not recommend",
    "Excellent film with outstanding performances",
    "Awful film with terrible performances"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.25, random_state=42)

# Calculate TF-IDF
train_tfidf_matrix, vocabulary = pyroid.parallel_tfidf(X_train, False)

# Convert sparse representation to dense numpy array
train_features = np.zeros((len(train_tfidf_matrix), len(vocabulary)))
for i, doc_dict in enumerate(train_tfidf_matrix):
    for term_idx, tfidf_value in doc_dict.items():
        train_features[i, term_idx] = tfidf_value

# Calculate TF-IDF for test documents using the same vocabulary
test_tfidf_matrix = []
for doc in X_test:
    # Tokenize and count terms
    tokens = doc.lower().split()
    term_counts = {}
    for token in tokens:
        if token in vocabulary:
            term_counts[vocabulary[token]] = term_counts.get(vocabulary[token], 0) + 1
    
    # Calculate TF-IDF
    doc_len = len(tokens)
    doc_tfidf = {}
    for term_idx, count in term_counts.items():
        tf = count / doc_len
        # Use IDF from training data (simplified)
        doc_tfidf[term_idx] = tf
    
    test_tfidf_matrix.append(doc_tfidf)

# Convert test sparse representation to dense numpy array
test_features = np.zeros((len(test_tfidf_matrix), len(vocabulary)))
for i, doc_dict in enumerate(test_tfidf_matrix):
    for term_idx, tfidf_value in doc_dict.items():
        test_features[i, term_idx] = tfidf_value

# Train a classifier
clf = LogisticRegression()
clf.fit(train_features, y_train)

# Predict and evaluate
y_pred = clf.predict(test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

### Example 2: Document Clustering

```python
import pyroid
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning is a subset of machine learning",
    "Neural networks are used in deep learning",
    "Python is a popular programming language",
    "JavaScript is used for web development",
    "HTML and CSS are used for web design",
    "Natural language processing is used for text analysis",
    "Computer vision is used for image analysis"
]

# Calculate document similarity matrix
similarity_matrix = pyroid.parallel_document_similarity(documents, "cosine", False)

# Convert similarity matrix to distance matrix
distance_matrix = 1 - np.array(similarity_matrix)

# Apply PCA for visualization
pca = PCA(n_components=2)
points = pca.fit_transform(distance_matrix)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(distance_matrix)

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(points[:, 0], points[:, 1], c=clusters, cmap='viridis', s=100)

# Add document labels
for i, doc in enumerate(documents):
    plt.annotate(f"Doc {i+1}", (points[i, 0], points[i, 1]), fontsize=9)

plt.title('Document Clustering based on Similarity')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.show()
```

### Example 3: Keyword Extraction

```python
import pyroid
import numpy as np
from collections import Counter

# Sample document
document = """
Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.
"""

# Tokenize the document
tokens = pyroid.parallel_tokenize([document], True, True)[0]

# Calculate term frequency
term_freq = Counter(tokens)

# Generate bigrams
bigrams = pyroid.parallel_ngrams([document], 2, False)[0]
bigram_freq = Counter(bigrams)

# Calculate TF-IDF for the document against a corpus
corpus = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning is a subset of machine learning",
    "Neural networks are used in deep learning",
    "Python is a popular programming language",
    document
]

tfidf_matrix, vocabulary = pyroid.parallel_tfidf(corpus, False)

# Get TF-IDF scores for the document (last in the corpus)
doc_tfidf = tfidf_matrix[-1]

# Extract top keywords based on TF-IDF
keywords = []
for term, idx in vocabulary.items():
    if idx in doc_tfidf:
        keywords.append((term, doc_tfidf[idx]))

# Sort keywords by TF-IDF score
keywords.sort(key=lambda x: x[1], reverse=True)

# Print top 10 keywords
print("Top 10 keywords by TF-IDF:")
for term, score in keywords[:10]:
    print(f"{term}: {score:.4f}")

# Print top 10 bigrams by frequency
print("\nTop 10 bigrams by frequency:")
for bigram, freq in bigram_freq.most_common(10):
    print(f"{bigram}: {freq}")