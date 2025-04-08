# Machine Learning Operations

The Machine Learning operations module provides basic machine learning algorithms implemented in pure Rust without external dependencies. These operations are designed to be simple, reliable, and easy to use.

## Overview

The Machine Learning operations module provides the following key functions:

- `kmeans`: K-means clustering algorithm
- `linear_regression`: Simple linear regression
- `normalize`: Data normalization
- `distance_matrix`: Calculate distance matrix between points

## API Reference

### kmeans

Perform K-means clustering on a dataset.

```python
pyroid.ml.basic.kmeans(data, k, max_iterations=None)
```

#### Parameters

- `data`: A list of points (each point is a list of coordinates)
- `k`: Number of clusters
- `max_iterations`: Maximum number of iterations (default: 100)

#### Returns

A dictionary containing:
- `centroids`: List of cluster centroids
- `clusters`: List of cluster assignments for each point
- `iterations`: Number of iterations performed

#### Example

```python
import pyroid
import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
np.random.seed(42)
data = []
# Cluster 1
for _ in range(50):
    data.append([np.random.normal(0, 1), np.random.normal(0, 1)])
# Cluster 2
for _ in range(50):
    data.append([np.random.normal(5, 1), np.random.normal(5, 1)])
# Cluster 3
for _ in range(50):
    data.append([np.random.normal(0, 1), np.random.normal(5, 1)])

# Perform K-means clustering
result = pyroid.ml.basic.kmeans(data, k=3)

# Extract results
centroids = result['centroids']
clusters = result['clusters']
iterations = result['iterations']

print(f"K-means converged in {iterations} iterations")
print(f"Centroids: {centroids}")

# Plot the results
data_np = np.array(data)
plt.figure(figsize=(10, 6))
plt.scatter(data_np[:, 0], data_np[:, 1], c=clusters, cmap='viridis')
centroids_np = np.array(centroids)
plt.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='X', s=100)
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

### linear_regression

Perform simple linear regression.

```python
pyroid.ml.basic.linear_regression(x, y)
```

#### Parameters

- `x`: List of independent variable values
- `y`: List of dependent variable values

#### Returns

A dictionary containing:
- `slope`: Slope of the regression line
- `intercept`: Y-intercept of the regression line
- `r_squared`: R-squared value (coefficient of determination)

#### Example

```python
import pyroid
import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
np.random.seed(42)
x = list(range(1, 101))
y = [2 * xi + 5 + np.random.normal(0, 10) for xi in x]

# Perform linear regression
result = pyroid.ml.basic.linear_regression(x, y)

# Extract results
slope = result['slope']
intercept = result['intercept']
r_squared = result['r_squared']

print(f"Linear regression: y = {slope:.4f}x + {intercept:.4f}")
print(f"R-squared: {r_squared:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.plot(x, [slope * xi + intercept for xi in x], 'r-')
plt.title(f'Linear Regression (y = {slope:.4f}x + {intercept:.4f}, R² = {r_squared:.4f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

### normalize

Normalize a vector of values using different methods.

```python
pyroid.ml.basic.normalize(values, method=None)
```

#### Parameters

- `values`: List of values to normalize
- `method`: Normalization method (default: 'minmax')
  - Supported methods: 'minmax', 'zscore'

#### Returns

A list of normalized values.

#### Example

```python
import pyroid
import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
np.random.seed(42)
values = [np.random.normal(50, 10) for _ in range(100)]

# Normalize using different methods
minmax_normalized = pyroid.ml.basic.normalize(values, method='minmax')
zscore_normalized = pyroid.ml.basic.normalize(values, method='zscore')

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.hist(values, bins=20)
plt.title('Original Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(minmax_normalized, bins=20)
plt.title('Min-Max Normalization')
plt.xlabel('Value')

plt.subplot(1, 3, 3)
plt.hist(zscore_normalized, bins=20)
plt.title('Z-Score Normalization')
plt.xlabel('Value')

plt.tight_layout()
plt.show()

# Print statistics
print(f"Original data - Min: {min(values):.2f}, Max: {max(values):.2f}, Mean: {sum(values)/len(values):.2f}")
print(f"Min-Max normalized - Min: {min(minmax_normalized):.2f}, Max: {max(minmax_normalized):.2f}")
print(f"Z-Score normalized - Mean: {sum(zscore_normalized)/len(zscore_normalized):.2f}, Std: {np.std(zscore_normalized):.2f}")
```

#### Normalization Methods

1. **Min-Max Normalization ('minmax')**

   Scales values to a range between 0 and 1:
   
   ```
   z = (x - min(x)) / (max(x) - min(x))
   ```

2. **Z-Score Normalization ('zscore')**

   Standardizes values to have mean 0 and standard deviation 1:
   
   ```
   z = (x - mean(x)) / std(x)
   ```

### distance_matrix

Calculate distance matrix between points.

```python
pyroid.ml.basic.distance_matrix(points, metric=None)
```

#### Parameters

- `points`: A list of points (each point is a list of coordinates)
- `metric`: Distance metric to use (default: 'euclidean')
  - Supported metrics: 'euclidean', 'manhattan'

#### Returns

A 2D list representing the distance matrix.

#### Example

```python
import pyroid
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Generate some sample data
np.random.seed(42)
points = []
for _ in range(10):
    points.append([np.random.random() * 10, np.random.random() * 10])

# Calculate distance matrices using different metrics
euclidean_distances = pyroid.ml.basic.distance_matrix(points, metric='euclidean')
manhattan_distances = pyroid.ml.basic.distance_matrix(points, metric='manhattan')

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(euclidean_distances, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0])
axes[0].set_title('Euclidean Distance Matrix')

sns.heatmap(manhattan_distances, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1])
axes[1].set_title('Manhattan Distance Matrix')

plt.tight_layout()
plt.show()
```

## Performance Considerations

- The current implementation is focused on simplicity and reliability rather than maximum performance.
- Operations are performed in pure Rust without external dependencies, which provides good baseline performance.
- For very large datasets, memory usage can be a concern.
- The implementation does not currently support parallel processing.

## Best Practices

1. **Choose the appropriate distance metric**: Different distance metrics are suitable for different types of data. For example, 'euclidean' is suitable for continuous data, while 'manhattan' may be better for sparse data.

2. **Normalize data before clustering**: K-means and other distance-based algorithms are sensitive to the scale of the features. Consider using `normalize` before applying clustering.

3. **Set appropriate number of clusters**: For K-means, choosing the right number of clusters is important. Consider using techniques like the elbow method or silhouette analysis.

4. **Consider convergence criteria**: K-means may not always converge to the global optimum. Try running the algorithm multiple times with different initializations.

## Limitations

1. **Basic algorithms only**: The current implementation provides only basic machine learning algorithms and does not include advanced features like regularization, kernel methods, or deep learning.

2. **Limited metrics**: Only a few distance metrics are supported.

3. **No parallel processing**: The current implementation does not support parallel processing.

4. **Memory usage**: For very large datasets, memory usage can be a concern.

## Examples

### Example 1: Finding Optimal Number of Clusters

```python
import pyroid
import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
np.random.seed(42)
data = []
# Cluster 1
for _ in range(50):
    data.append([np.random.normal(0, 1), np.random.normal(0, 1)])
# Cluster 2
for _ in range(50):
    data.append([np.random.normal(5, 1), np.random.normal(5, 1)])
# Cluster 3
for _ in range(50):
    data.append([np.random.normal(0, 1), np.random.normal(5, 1)])

# Try different numbers of clusters
inertias = []
for k in range(1, 11):
    result = pyroid.ml.basic.kmeans(data, k=k)
    
    # Calculate inertia (sum of squared distances to nearest centroid)
    inertia = 0
    centroids = result['centroids']
    clusters = result['clusters']
    
    for i, point in enumerate(data):
        centroid = centroids[clusters[i]]
        # Calculate squared distance
        dist_sq = sum((point[j] - centroid[j])**2 for j in range(len(point)))
        inertia += dist_sq
    
    inertias.append(inertia)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()
```

### Example 2: Predicting with Linear Regression

```python
import pyroid
import matplotlib.pyplot as plt
import numpy as np

# Generate training data
np.random.seed(42)
x_train = list(range(1, 101))
y_train = [3 * xi + 2 + np.random.normal(0, 5) for xi in x_train]

# Perform linear regression
result = pyroid.ml.basic.linear_regression(x_train, y_train)
slope = result['slope']
intercept = result['intercept']
r_squared = result['r_squared']

# Generate test data
x_test = list(range(101, 121))
y_test = [3 * xi + 2 + np.random.normal(0, 5) for xi in x_test]

# Make predictions
y_pred = [slope * xi + intercept for xi in x_test]

# Calculate mean squared error
mse = sum((y_test[i] - y_pred[i])**2 for i in range(len(y_test))) / len(y_test)
print(f"Mean Squared Error on test data: {mse:.2f}")

# Plot the results
plt.figure(figsize=(12, 6))

# Training data and regression line
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, alpha=0.5, label='Training Data')
plt.plot(x_train, [slope * xi + intercept for xi in x_train], 'r-', label=f'Regression Line (y = {slope:.2f}x + {intercept:.2f})')
plt.title(f'Linear Regression (R² = {r_squared:.4f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Test data and predictions
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_test, alpha=0.5, label='Test Data')
plt.scatter(x_test, y_pred, color='red', marker='x', label='Predictions')
plt.title(f'Predictions on Test Data (MSE = {mse:.2f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()