# Machine Learning Operations

This document provides detailed information about the machine learning operations available in Pyroid.

## K-means Clustering

```python
pyroid.ml.basic.kmeans(data, k, max_iterations=100, tolerance=1e-4)
```

Performs k-means clustering on the input data.

**Parameters:**
- `data` (list): A list of data points, where each data point is a list of features.
- `k` (int): The number of clusters to form.
- `max_iterations` (int, optional): The maximum number of iterations. Default is 100.
- `tolerance` (float, optional): The convergence tolerance. Default is 1e-4.

**Returns:**
- A dictionary containing:
  - `centroids`: A list of cluster centroids.
  - `clusters`: A list of cluster assignments for each data point.
  - `iterations`: The number of iterations performed.

**Example:**
```python
import pyroid

# Sample data
data = [
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
    [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
]

# Perform k-means clustering
result = pyroid.ml.basic.kmeans(data, k=2)

print(f"Centroids: {result['centroids']}")
print(f"Clusters: {result['clusters']}")
print(f"Iterations: {result['iterations']}")
```

## Linear Regression

```python
pyroid.ml.basic.linear_regression(X, y)
```

Performs linear regression on the input data.

**Parameters:**
- `X` (list): A list of feature vectors, where each feature vector is a list of features. For simple linear regression, each feature vector can be a single value.
- `y` (list): A list of target values.

**Returns:**
- A dictionary containing:
  - `coefficients`: The coefficients of the linear regression model.
  - `intercept`: The intercept of the linear regression model.
  - `r_squared`: The coefficient of determination (R²).

**Example:**
```python
import pyroid

# Sample data for multiple linear regression
X = [[1, 1], [1, 2], [2, 2], [2, 3]]
y = [6, 8, 9, 11]

# Perform linear regression
result = pyroid.ml.basic.linear_regression(X, y)

print(f"Coefficients: {result['coefficients']}")
print(f"Intercept: {result['intercept']}")
print(f"R-squared: {result['r_squared']}")
```

## Data Normalization

```python
pyroid.ml.basic.normalize(data, method="min-max")
```

Normalizes the input data using the specified method.

**Parameters:**
- `data` (list): A list of data points or a list of feature vectors.
- `method` (str, optional): The normalization method. One of "min-max", "z-score". Default is "min-max".

**Returns:**
- A list of normalized data points.

**Example:**
```python
import pyroid

# Sample data
data = [10.0, 20.0, 30.0, 40.0, 50.0]

# Normalize using min-max scaling
min_max_normalized = pyroid.ml.basic.normalize(data, method="min-max")
print(f"Min-Max Normalized: {min_max_normalized}")

# Normalize using z-score scaling
z_score_normalized = pyroid.ml.basic.normalize(data, method="z-score")
print(f"Z-Score Normalized: {z_score_normalized}")
```

## Distance Matrix

```python
pyroid.ml.basic.distance_matrix(data, metric="euclidean")
```

Computes the distance matrix for the input data using the specified metric.

**Parameters:**
- `data` (list): A list of data points, where each data point is a list of features.
- `metric` (str, optional): The distance metric. One of "euclidean", "manhattan", "cosine". Default is "euclidean".

**Returns:**
- A 2D list representing the distance matrix.

**Example:**
```python
import pyroid

# Sample data
data = [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
]

# Compute Euclidean distance matrix
euclidean_distances = pyroid.ml.basic.distance_matrix(data, metric="euclidean")
print("Euclidean Distance Matrix:")
for row in euclidean_distances:
    print(row)

# Compute Manhattan distance matrix
manhattan_distances = pyroid.ml.basic.distance_matrix(data, metric="manhattan")
print("\nManhattan Distance Matrix:")
for row in manhattan_distances:
    print(row)
```

## Principal Component Analysis (PCA)

```python
pyroid.ml.basic.pca(data, n_components=None)
```

Performs Principal Component Analysis (PCA) on the input data.

**Parameters:**
- `data` (list): A list of data points, where each data point is a list of features.
- `n_components` (int, optional): The number of components to keep. If None, all components are kept. Default is None.

**Returns:**
- A dictionary containing:
  - `components`: The principal components.
  - `explained_variance`: The explained variance of each component.
  - `transformed_data`: The data transformed into the new space.

**Example:**
```python
import pyroid

# Sample data
data = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
]

# Perform PCA
result = pyroid.ml.basic.pca(data, n_components=2)

print(f"Components: {result['components']}")
print(f"Explained Variance: {result['explained_variance']}")
print(f"Transformed Data: {result['transformed_data']}")
```

## K-Nearest Neighbors

```python
pyroid.ml.basic.knn_classify(train_data, train_labels, test_data, k=3, metric="euclidean")
```

Performs k-nearest neighbors classification.

**Parameters:**
- `train_data` (list): A list of training data points, where each data point is a list of features.
- `train_labels` (list): A list of labels for the training data.
- `test_data` (list): A list of test data points to classify.
- `k` (int, optional): The number of neighbors to consider. Default is 3.
- `metric` (str, optional): The distance metric. One of "euclidean", "manhattan", "cosine". Default is "euclidean".

**Returns:**
- A list of predicted labels for the test data.

**Example:**
```python
import pyroid

# Sample data
train_data = [
    [1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
    [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]
]
train_labels = [0, 0, 0, 1, 1, 1]
test_data = [
    [1.5, 2.5],
    [5.5, 6.5]
]

# Perform KNN classification
predictions = pyroid.ml.basic.knn_classify(train_data, train_labels, test_data, k=3)
print(f"Predictions: {predictions}")
```

## Decision Tree

```python
pyroid.ml.basic.decision_tree(train_data, train_labels, max_depth=None)
```

Trains a decision tree classifier.

**Parameters:**
- `train_data` (list): A list of training data points, where each data point is a list of features.
- `train_labels` (list): A list of labels for the training data.
- `max_depth` (int, optional): The maximum depth of the tree. If None, the tree is grown until all leaves are pure. Default is None.

**Returns:**
- A decision tree model that can be used for prediction.

**Example:**
```python
import pyroid

# Sample data
train_data = [
    [1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
    [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]
]
train_labels = [0, 0, 0, 1, 1, 1]
test_data = [
    [1.5, 2.5],
    [5.5, 6.5]
]

# Train a decision tree
tree = pyroid.ml.basic.decision_tree(train_data, train_labels, max_depth=3)

# Make predictions
predictions = tree.predict(test_data)
print(f"Predictions: {predictions}")
```

## Naive Bayes

```python
pyroid.ml.basic.naive_bayes(train_data, train_labels)
```

Trains a Gaussian Naive Bayes classifier.

**Parameters:**
- `train_data` (list): A list of training data points, where each data point is a list of features.
- `train_labels` (list): A list of labels for the training data.

**Returns:**
- A Naive Bayes model that can be used for prediction.

**Example:**
```python
import pyroid

# Sample data
train_data = [
    [1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
    [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]
]
train_labels = [0, 0, 0, 1, 1, 1]
test_data = [
    [1.5, 2.5],
    [5.5, 6.5]
]

# Train a Naive Bayes classifier
nb = pyroid.ml.basic.naive_bayes(train_data, train_labels)

# Make predictions
predictions = nb.predict(test_data)
print(f"Predictions: {predictions}")
```

## Support Vector Machine (SVM)

```python
pyroid.ml.basic.svm(train_data, train_labels, kernel="linear", C=1.0)
```

Trains a Support Vector Machine classifier.

**Parameters:**
- `train_data` (list): A list of training data points, where each data point is a list of features.
- `train_labels` (list): A list of labels for the training data.
- `kernel` (str, optional): The kernel type. One of "linear", "poly", "rbf", "sigmoid". Default is "linear".
- `C` (float, optional): The regularization parameter. Default is 1.0.

**Returns:**
- An SVM model that can be used for prediction.

**Example:**
```python
import pyroid

# Sample data
train_data = [
    [1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
    [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]
]
train_labels = [0, 0, 0, 1, 1, 1]
test_data = [
    [1.5, 2.5],
    [5.5, 6.5]
]

# Train an SVM classifier
svm = pyroid.ml.basic.svm(train_data, train_labels, kernel="linear", C=1.0)

# Make predictions
predictions = svm.predict(test_data)
print(f"Predictions: {predictions}")
```

## Random Forest

```python
pyroid.ml.basic.random_forest(train_data, train_labels, n_trees=100, max_depth=None)
```

Trains a Random Forest classifier.

**Parameters:**
- `train_data` (list): A list of training data points, where each data point is a list of features.
- `train_labels` (list): A list of labels for the training data.
- `n_trees` (int, optional): The number of trees in the forest. Default is 100.
- `max_depth` (int, optional): The maximum depth of each tree. If None, the trees are grown until all leaves are pure. Default is None.

**Returns:**
- A Random Forest model that can be used for prediction.

**Example:**
```python
import pyroid

# Sample data
train_data = [
    [1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
    [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]
]
train_labels = [0, 0, 0, 1, 1, 1]
test_data = [
    [1.5, 2.5],
    [5.5, 6.5]
]

# Train a Random Forest classifier
rf = pyroid.ml.basic.random_forest(train_data, train_labels, n_trees=10, max_depth=3)

# Make predictions
predictions = rf.predict(test_data)
print(f"Predictions: {predictions}")