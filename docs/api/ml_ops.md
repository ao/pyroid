# Machine Learning Operations

The Machine Learning operations module provides high-performance implementations of common machine learning primitives and utilities. These operations are implemented in Rust and are designed to be significantly faster than their Python equivalents, especially for large datasets.

## Overview

The Machine Learning operations module provides the following key functions:

- `parallel_distance_matrix`: Calculate distance matrices in parallel
- `parallel_feature_scaling`: Scale features using various methods
- `parallel_cross_validation`: Perform cross-validation in parallel

## API Reference

### parallel_distance_matrix

Calculate distance matrix in parallel.

```python
pyroid.parallel_distance_matrix(points, metric='euclidean')
```

#### Parameters

- `points`: A list of points (each point is a list of coordinates)
- `metric`: Distance metric to use (default: 'euclidean')
  - Supported metrics: 'euclidean', 'manhattan', 'cosine'

#### Returns

A 2D array of distances between points.

#### Example

```python
import pyroid
import numpy as np
import time

# Generate random points
n_points = 2000
n_dims = 10
points = [[np.random.random() for _ in range(n_dims)] for _ in range(n_points)]
points_np = np.array(points)

# Calculate distance matrix using NumPy
def numpy_distance_matrix(points):
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

# Compare performance
start = time.time()
numpy_result = numpy_distance_matrix(points_np)
numpy_time = time.time() - start

start = time.time()
pyroid_result = pyroid.parallel_distance_matrix(points, "euclidean")
pyroid_time = time.time() - start

print(f"NumPy time: {numpy_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {numpy_time / pyroid_time:.1f}x")
```

#### Performance Considerations

- `parallel_distance_matrix` is particularly efficient for large datasets with many points.
- The implementation uses Rayon for parallel computation, which can lead to significant performance improvements on multi-core systems.
- The 'cosine' metric includes an optimization to precompute norms, which makes it faster than a naive implementation.
- For very large datasets, memory usage can be a concern as the distance matrix grows quadratically with the number of points.

### parallel_feature_scaling

Scale features in parallel.

```python
pyroid.parallel_feature_scaling(data, method='standard', with_mean=True, with_std=True)
```

#### Parameters

- `data`: A 2D array of data (rows are samples, columns are features)
- `method`: Scaling method (default: 'standard')
  - Supported methods: 'standard', 'minmax', 'robust'
- `with_mean`: Whether to center the data before scaling (default: True)
- `with_std`: Whether to scale to unit variance (default: True)

#### Returns

A 2D array of scaled data.

#### Example

```python
import pyroid
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

# Generate random data
n_samples = 100000
n_features = 20
data = [[np.random.randn() * 10 for _ in range(n_features)] for _ in range(n_samples)]
data_np = np.array(data)

# Compare with scikit-learn
start = time.time()
scaler = StandardScaler()
sklearn_result = scaler.fit_transform(data_np)
sklearn_time = time.time() - start

start = time.time()
pyroid_result = pyroid.parallel_feature_scaling(data, "standard")
pyroid_time = time.time() - start

print(f"Scikit-learn time: {sklearn_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {sklearn_time / pyroid_time:.1f}x")
```

#### Scaling Methods

1. **Standard Scaling ('standard')**

   Standardizes features by removing the mean and scaling to unit variance:
   
   ```
   z = (x - μ) / σ
   ```
   
   where μ is the mean and σ is the standard deviation of the feature.

2. **Min-Max Scaling ('minmax')**

   Scales features to a given range (default [0, 1]):
   
   ```
   z = (x - min(x)) / (max(x) - min(x))
   ```

3. **Robust Scaling ('robust')**

   Scales features using statistics that are robust to outliers:
   
   ```
   z = (x - median(x)) / IQR(x)
   ```
   
   where IQR(x) is the interquartile range (Q3 - Q1).

#### Performance Considerations

- `parallel_feature_scaling` is particularly efficient for large datasets with many samples and features.
- The implementation processes each feature in parallel, which can lead to significant performance improvements on multi-core systems.
- The 'standard' and 'minmax' methods are generally faster than the 'robust' method, as the latter requires sorting the data to compute the median and quartiles.

### parallel_cross_validation

Perform cross-validation in parallel.

```python
pyroid.parallel_cross_validation(X, y, cv=5, model_func=None, scoring_func=None)
```

#### Parameters

- `X`: A 2D array of features (rows are samples, columns are features)
- `y`: A 1D array of target values
- `cv`: Number of folds (default: 5)
- `model_func`: A Python function that takes (X_train, y_train, X_test) and returns predictions
- `scoring_func`: A Python function that takes (y_true, y_pred) and returns a score

#### Returns

A list of scores for each fold.

#### Example

```python
import pyroid
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

# Generate a classification dataset
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_list = X.tolist()
y_list = y.tolist()

# Define a simple model function for pyroid
def knn_model(X_train, y_train, X_test):
    # Convert to numpy arrays
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    
    # Train KNN model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_np, y_train_np)
    
    # Predict
    return model.predict(X_test_np).tolist()

# Define scoring function
def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)

# Compare with scikit-learn
start = time.time()
sklearn_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), X, y, cv=5)
sklearn_time = time.time() - start

start = time.time()
pyroid_scores = pyroid.parallel_cross_validation(X_list, y_list, 5, knn_model, accuracy)
pyroid_time = time.time() - start

print(f"Scikit-learn time: {sklearn_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {sklearn_time / pyroid_time:.1f}x")
print(f"Scikit-learn scores: {sklearn_scores}")
print(f"Pyroid scores: {pyroid_scores}")
```

#### Performance Considerations

- `parallel_cross_validation` is particularly efficient for large datasets and computationally intensive models.
- The implementation processes each fold in parallel, which can lead to significant performance improvements on multi-core systems.
- The performance gain depends on the complexity of the model and the size of the dataset. For simple models and small datasets, the overhead of parallelization may outweigh the benefits.
- The function allows you to use any model and scoring function, making it highly flexible.

## Performance Comparison

The following table shows the performance comparison between scikit-learn and pyroid for various machine learning operations:

| Operation | Dataset Size | scikit-learn | pyroid | Speedup |
|-----------|-------------|-------------|--------|---------|
| Distance Matrix | 2000 points, 10 dims | 2500ms | 200ms | 12.5x |
| Standard Scaling | 100K samples, 20 features | 800ms | 100ms | 8.0x |
| Cross-Validation (KNN) | 10K samples, 20 features | 3000ms | 500ms | 6.0x |

## Best Practices

1. **Choose the appropriate distance metric**: Different distance metrics are suitable for different types of data. For example, 'euclidean' is suitable for continuous data, while 'manhattan' may be better for sparse data.

2. **Scale features before distance calculations**: Distance metrics are sensitive to the scale of the features. Consider using `parallel_feature_scaling` before calculating distances.

3. **Be mindful of memory usage**: Distance matrices grow quadratically with the number of points. For very large datasets, consider using alternative approaches or processing data in chunks.

4. **Use appropriate scaling method**: Choose the scaling method based on your data characteristics. For data with outliers, 'robust' scaling may be more appropriate than 'standard' scaling.

5. **Customize model and scoring functions**: The `parallel_cross_validation` function allows you to use any model and scoring function. Take advantage of this flexibility to tailor the cross-validation to your specific needs.

## Limitations

1. **Memory usage**: For very large datasets, memory usage can be a concern, especially for distance matrices.

2. **Limited set of distance metrics**: Currently, only 'euclidean', 'manhattan', and 'cosine' distance metrics are supported.

3. **No sparse matrix support**: The current implementation does not support sparse matrices, which may be inefficient for sparse data.

4. **Model function overhead**: When using `parallel_cross_validation`, there is some overhead in converting between Python and Rust data structures, which may reduce the performance gain for very simple models.

## Examples

### Example 1: Clustering with Distance Matrix

```python
import pyroid
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Generate random points in 2D space
n_points = 500
points = [[np.random.random() * 10, np.random.random() * 10] for _ in range(n_points)]

# Calculate distance matrix
distances = pyroid.parallel_distance_matrix(points, "euclidean")

# Convert to numpy array for scikit-learn
distances_np = np.array(distances)

# Perform hierarchical clustering
clustering = AgglomerativeClustering(
    n_clusters=5,
    affinity='precomputed',
    linkage='average'
).fit(distances_np)

# Plot the results
points_np = np.array(points)
plt.figure(figsize=(10, 8))
plt.scatter(points_np[:, 0], points_np[:, 1], c=clustering.labels_, cmap='viridis')
plt.title('Hierarchical Clustering with Pyroid Distance Matrix')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()
```

### Example 2: Feature Preprocessing Pipeline

```python
import pyroid
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load wine dataset
wine = load_wine()
X, y = wine.data, wine.target
X_list = X.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_list, y, test_size=0.2, random_state=42)

# Scale features using different methods
X_train_standard = pyroid.parallel_feature_scaling(X_train, "standard")
X_train_minmax = pyroid.parallel_feature_scaling(X_train, "minmax")
X_train_robust = pyroid.parallel_feature_scaling(X_train, "robust")

X_test_standard = pyroid.parallel_feature_scaling(X_test, "standard")
X_test_minmax = pyroid.parallel_feature_scaling(X_test, "minmax")
X_test_robust = pyroid.parallel_feature_scaling(X_test, "robust")

# Train and evaluate models with different scaling methods
for name, X_tr, X_te in [
    ("Standard", X_train_standard, X_test_standard),
    ("MinMax", X_train_minmax, X_test_minmax),
    ("Robust", X_train_robust, X_test_robust)
]:
    clf = RandomForestClassifier(random_state=42)
    clf.fit(np.array(X_tr), y_train)
    y_pred = clf.predict(np.array(X_te))
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} scaling accuracy: {acc:.4f}")
```

### Example 3: Custom Cross-Validation

```python
import pyroid
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load breast cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_list = X.tolist()
y_list = y.tolist()

# Scale features
X_scaled = pyroid.parallel_feature_scaling(X_list, "standard")

# Define model function with regularization strength as a parameter
def logistic_regression_model(X_train, y_train, X_test, C=1.0):
    # Convert to numpy arrays
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    
    # Train logistic regression model
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_train_np, y_train_np)
    
    # Predict probabilities
    return model.predict_proba(X_test_np)[:, 1].tolist()

# Define scoring function (AUC)
def auc_score(y_true, y_prob):
    # Simple AUC calculation
    pos = [i for i, y in enumerate(y_true) if y == 1]
    neg = [i for i, y in enumerate(y_true) if y == 0]
    
    n_pos = len(pos)
    n_neg = len(neg)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Count concordant pairs
    concordant = 0
    for i in pos:
        for j in neg:
            if y_prob[i] > y_prob[j]:
                concordant += 1
            elif y_prob[i] == y_prob[j]:
                concordant += 0.5
    
    return concordant / (n_pos * n_neg)

# Test different regularization strengths
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    # Create a closure with the current C value
    model_func = lambda X_tr, y_tr, X_te, C=C: logistic_regression_model(X_tr, y_tr, X_te, C)
    
    # Perform cross-validation
    scores = pyroid.parallel_cross_validation(X_scaled, y_list, 5, model_func, auc_score)
    
    print(f"C={C}, Mean AUC: {sum(scores) / len(scores):.4f}, Scores: {[f'{s:.4f}' for s in scores]}")