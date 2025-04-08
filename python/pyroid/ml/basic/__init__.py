"""
Pyroid Machine Learning Basic Module
================================

This module provides basic machine learning operations.

Functions:
    kmeans: K-means clustering
    linear_regression: Linear regression
    normalize: Normalize data
    distance_matrix: Calculate distance matrix
"""

# Try to import directly from the pyroid module
try:
    from ...pyroid import (
        # Clustering
        kmeans,
        
        # Regression
        linear_regression,
        
        # Data preprocessing
        normalize,
        
        # Distance calculations
        distance_matrix,
    )
except ImportError:
    # Fallback to importing from the ml module
    try:
        from ...pyroid.ml import (
            # Clustering
            kmeans,
            
            # Regression
            linear_regression,
            
            # Data preprocessing
            normalize,
            
            # Distance calculations
            distance_matrix,
        )
    except ImportError:
        # If all else fails, create dummy functions for documentation purposes
        import math
        import random
        
        def kmeans(data, k=2, max_iterations=100):
            """K-means clustering (fallback implementation)."""
            if not data or k <= 0 or k > len(data):
                return {"centroids": [], "clusters": []}
                
            # Initialize centroids randomly
            centroids = random.sample(data, k)
            
            for _ in range(max_iterations):
                # Assign points to clusters
                clusters = [[] for _ in range(k)]
                for point in data:
                    closest_centroid = min(range(k), key=lambda i: _distance(point, centroids[i]))
                    clusters[closest_centroid].append(point)
                
                # Update centroids
                new_centroids = []
                for i, cluster in enumerate(clusters):
                    if not cluster:
                        new_centroids.append(centroids[i])
                        continue
                    
                    # Calculate mean of cluster
                    dim = len(cluster[0])
                    new_centroid = [0] * dim
                    for point in cluster:
                        for j in range(dim):
                            new_centroid[j] += point[j]
                    new_centroid = [coord / len(cluster) for coord in new_centroid]
                    new_centroids.append(new_centroid)
                
                # Check for convergence
                if _centroids_equal(centroids, new_centroids):
                    break
                    
                centroids = new_centroids
                
            return {
                "centroids": centroids,
                "clusters": clusters
            }
            
        def linear_regression(X, y):
            """Linear regression (fallback implementation)."""
            if not X or not y or len(X) != len(y):
                return {"coefficients": [], "intercept": 0, "r_squared": 0}
                
            # Add intercept term
            X_with_intercept = [[1] + x for x in X]
            
            # Calculate coefficients using normal equation
            X_transpose = _transpose(X_with_intercept)
            X_transpose_X = _matrix_multiply(X_transpose, X_with_intercept)
            X_transpose_y = _matrix_vector_multiply(X_transpose, y)
            
            try:
                coeffs = _solve_linear_system(X_transpose_X, X_transpose_y)
            except:
                return {"coefficients": [0] * len(X[0]), "intercept": 0, "r_squared": 0}
                
            intercept = coeffs[0]
            coefficients = coeffs[1:]
            
            # Calculate R-squared
            y_mean = sum(y) / len(y)
            ss_total = sum((yi - y_mean) ** 2 for yi in y)
            y_pred = [_dot_product([1] + x, coeffs) for x in X]
            ss_residual = sum((yi - y_pred_i) ** 2 for yi, y_pred_i in zip(y, y_pred))
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            return {
                "coefficients": coefficients,
                "intercept": intercept,
                "r_squared": r_squared
            }
            
        def normalize(data, method="min-max"):
            """Normalize data (fallback implementation)."""
            if not data:
                return []
                
            # Transpose to get columns
            columns = _transpose(data)
            normalized_columns = []
            
            for column in columns:
                if method == "min-max":
                    min_val = min(column)
                    max_val = max(column)
                    if max_val == min_val:
                        normalized_columns.append([0.5] * len(column))
                    else:
                        normalized_columns.append([(x - min_val) / (max_val - min_val) for x in column])
                elif method == "z-score":
                    mean = sum(column) / len(column)
                    std_dev = math.sqrt(sum((x - mean) ** 2 for x in column) / len(column))
                    if std_dev == 0:
                        normalized_columns.append([0] * len(column))
                    else:
                        normalized_columns.append([(x - mean) / std_dev for x in column])
                else:
                    normalized_columns.append(column)  # No normalization
            
            # Transpose back to original format
            return _transpose(normalized_columns)
            
        def distance_matrix(data, metric="euclidean"):
            """Calculate distance matrix (fallback implementation)."""
            if not data:
                return []
                
            n = len(data)
            matrix = [[0] * n for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if metric == "euclidean":
                        dist = _distance(data[i], data[j])
                    elif metric == "manhattan":
                        dist = sum(abs(a - b) for a, b in zip(data[i], data[j]))
                    elif metric == "cosine":
                        dot = _dot_product(data[i], data[j])
                        norm_i = math.sqrt(_dot_product(data[i], data[i]))
                        norm_j = math.sqrt(_dot_product(data[j], data[j]))
                        dist = 1 - (dot / (norm_i * norm_j)) if norm_i * norm_j != 0 else 1
                    else:
                        dist = _distance(data[i], data[j])  # Default to euclidean
                        
                    matrix[i][j] = dist
                    matrix[j][i] = dist  # Distance matrix is symmetric
            
            return matrix
            
        # Helper functions
        def _distance(p1, p2):
            """Calculate Euclidean distance between two points."""
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            
        def _centroids_equal(c1, c2, tolerance=1e-6):
            """Check if two sets of centroids are equal within tolerance."""
            if len(c1) != len(c2):
                return False
            return all(_distance(a, b) < tolerance for a, b in zip(c1, c2))
            
        def _transpose(matrix):
            """Transpose a matrix."""
            if not matrix:
                return []
            return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
            
        def _dot_product(v1, v2):
            """Calculate dot product of two vectors."""
            return sum(a * b for a, b in zip(v1, v2))
            
        def _matrix_multiply(A, B):
            """Multiply two matrices."""
            B_transpose = _transpose(B)
            return [[_dot_product(row, col) for col in B_transpose] for row in A]
            
        def _matrix_vector_multiply(A, v):
            """Multiply a matrix by a vector."""
            return [_dot_product(row, v) for row in A]
            
        def _solve_linear_system(A, b):
            """Solve a linear system Ax = b using Gaussian elimination."""
            n = len(A)
            augmented = [row[:] + [b[i]] for i, row in enumerate(A)]
            
            # Gaussian elimination
            for i in range(n):
                # Find pivot
                max_row = i
                for j in range(i + 1, n):
                    if abs(augmented[j][i]) > abs(augmented[max_row][i]):
                        max_row = j
                
                # Swap rows
                augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
                
                # Check for singular matrix
                if abs(augmented[i][i]) < 1e-10:
                    raise ValueError("Matrix is singular")
                
                # Eliminate below
                for j in range(i + 1, n):
                    factor = augmented[j][i] / augmented[i][i]
                    for k in range(i, n + 1):
                        augmented[j][k] -= factor * augmented[i][k]
            
            # Back substitution
            x = [0] * n
            for i in range(n - 1, -1, -1):
                x[i] = augmented[i][n]
                for j in range(i + 1, n):
                    x[i] -= augmented[i][j] * x[j]
                x[i] /= augmented[i][i]
            
            return x

__all__ = [
    'kmeans',
    'linear_regression',
    'normalize',
    'distance_matrix',
]