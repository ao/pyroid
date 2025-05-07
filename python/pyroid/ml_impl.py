"""
Pyroid Machine Learning Implementation
=================================

This module provides Python implementations of the machine learning functions.

Functions:
    kmeans: K-means clustering
    linear_regression: Linear regression
    normalize: Normalize data
    distance_matrix: Calculate distance matrix
"""

import math
import random
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

def kmeans(data: List[List[float]], k: int, max_iterations: int = 100) -> Dict[str, Any]:
    """
    K-means clustering.
    
    Args:
        data: The data points
        k: The number of clusters
        max_iterations: The maximum number of iterations
        
    Returns:
        A dictionary with centroids and clusters
    """
    if not data:
        return {"centroids": [], "clusters": []}
    
    # Initialize centroids randomly
    centroids = random.sample(data, k)
    
    for _ in range(max_iterations):
        # Assign points to clusters
        clusters = [[] for _ in range(k)]
        
        for point in data:
            # Find the closest centroid
            closest_centroid = 0
            min_distance = float('inf')
            
            for i, centroid in enumerate(centroids):
                distance = euclidean_distance(point, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = i
            
            # Add the point to the cluster
            clusters[closest_centroid].append(point)
        
        # Update centroids
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                # If a cluster is empty, keep the old centroid
                new_centroids.append(centroids[i])
            else:
                # Calculate the mean of the cluster
                new_centroid = [0.0] * len(cluster[0])
                for point in cluster:
                    for j in range(len(point)):
                        new_centroid[j] += point[j]
                
                for j in range(len(new_centroid)):
                    new_centroid[j] /= len(cluster)
                
                new_centroids.append(new_centroid)
        
        # Check for convergence
        if centroids == new_centroids:
            break
        
        centroids = new_centroids
    
    return {"centroids": centroids, "clusters": clusters}

def linear_regression(X: List[List[float]], y: List[float]) -> Dict[str, Any]:
    """
    Linear regression.
    
    Args:
        X: The feature matrix
        y: The target vector
        
    Returns:
        A dictionary with coefficients, intercept, and R-squared
        
    Raises:
        ValueError: If X and y have different lengths
    """
    if not X or not y:
        return {"coefficients": [], "intercept": 0.0, "r_squared": 0.0}
    
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    
    n = len(X)
    p = len(X[0])
    
    # Add a column of ones for the intercept
    X_with_intercept = [[1.0] + row for row in X]
    
    # Calculate the coefficients using the normal equation
    # (X^T X)^(-1) X^T y
    
    # Calculate X^T X
    XT_X = [[0.0] * (p + 1) for _ in range(p + 1)]
    for i in range(p + 1):
        for j in range(p + 1):
            for k in range(n):
                XT_X[i][j] += X_with_intercept[k][i] * X_with_intercept[k][j]
    
    # Calculate X^T y
    XT_y = [0.0] * (p + 1)
    for i in range(p + 1):
        for k in range(n):
            XT_y[i] += X_with_intercept[k][i] * y[k]
    
    # Calculate (X^T X)^(-1)
    # For simplicity, we'll use a simple implementation for small matrices
    # In practice, you would use a more robust method
    XT_X_inv = inverse_matrix(XT_X)
    
    # Calculate (X^T X)^(-1) X^T y
    coefficients = [0.0] * (p + 1)
    for i in range(p + 1):
        for j in range(p + 1):
            coefficients[i] += XT_X_inv[i][j] * XT_y[j]
    
    # Extract the intercept and coefficients
    intercept = coefficients[0]
    coefficients = coefficients[1:]
    
    # Calculate R-squared
    y_mean = sum(y) / n
    ss_total = sum((yi - y_mean) ** 2 for yi in y)
    
    y_pred = [intercept + sum(coef * xi for coef, xi in zip(coefficients, X[i])) for i in range(n)]
    ss_residual = sum((yi - y_pred[i]) ** 2 for i, yi in enumerate(y))
    
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
    
    return {"coefficients": coefficients, "intercept": intercept, "r_squared": r_squared}

def normalize(data: List[List[float]], method: str = "min-max") -> List[List[float]]:
    """
    Normalize data.
    
    Args:
        data: The data to normalize
        method: The normalization method ("min-max" or "z-score")
        
    Returns:
        The normalized data
    """
    if not data:
        return []
    
    n = len(data)
    p = len(data[0])
    
    # Calculate statistics for each feature
    stats = []
    for j in range(p):
        values = [data[i][j] for i in range(n)]
        min_val = min(values)
        max_val = max(values)
        mean_val = sum(values) / n
        std_val = math.sqrt(sum((x - mean_val) ** 2 for x in values) / n)
        
        stats.append({
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val
        })
    
    # Normalize the data
    result = [[0.0] * p for _ in range(n)]
    
    for i in range(n):
        for j in range(p):
            if method == "min-max":
                # Min-max normalization
                min_val = stats[j]["min"]
                max_val = stats[j]["max"]
                
                if max_val == min_val:
                    result[i][j] = 0.0
                else:
                    result[i][j] = (data[i][j] - min_val) / (max_val - min_val)
            
            elif method == "z-score":
                # Z-score normalization
                mean_val = stats[j]["mean"]
                std_val = stats[j]["std"]
                
                if std_val == 0:
                    result[i][j] = 0.0
                else:
                    result[i][j] = (data[i][j] - mean_val) / std_val
            
            else:
                raise ValueError(f"Unknown normalization method: {method}")
    
    return result

def distance_matrix(data: List[List[float]], metric: str = "euclidean") -> List[List[float]]:
    """
    Calculate distance matrix.
    
    Args:
        data: The data points
        metric: The distance metric ("euclidean" or "manhattan")
        
    Returns:
        The distance matrix
    """
    if not data:
        return []
    
    n = len(data)
    result = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                result[i][j] = 0.0
            else:
                if metric == "euclidean":
                    distance = euclidean_distance(data[i], data[j])
                elif metric == "manhattan":
                    distance = manhattan_distance(data[i], data[j])
                else:
                    raise ValueError(f"Unknown distance metric: {metric}")
                
                result[i][j] = distance
                result[j][i] = distance
    
    return result

def euclidean_distance(a: List[float], b: List[float]) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        a: The first point
        b: The second point
        
    Returns:
        The Euclidean distance
    """
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def manhattan_distance(a: List[float], b: List[float]) -> float:
    """
    Calculate the Manhattan distance between two points.
    
    Args:
        a: The first point
        b: The second point
        
    Returns:
        The Manhattan distance
    """
    return sum(abs(ai - bi) for ai, bi in zip(a, b))

def inverse_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """
    Calculate the inverse of a matrix.
    
    Args:
        matrix: The matrix to invert
        
    Returns:
        The inverse matrix
    """
    n = len(matrix)
    
    # Create an augmented matrix [A|I]
    augmented = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(matrix)]
    
    # Gaussian elimination
    for i in range(n):
        # Find the pivot
        pivot = augmented[i][i]
        if pivot == 0:
            # Find a non-zero pivot
            for j in range(i + 1, n):
                if augmented[j][i] != 0:
                    augmented[i], augmented[j] = augmented[j], augmented[i]
                    break
            else:
                raise ValueError("Matrix is singular")
            
            pivot = augmented[i][i]
        
        # Scale the pivot row
        for j in range(i, 2 * n):
            augmented[i][j] /= pivot
        
        # Eliminate other rows
        for j in range(n):
            if j != i:
                factor = augmented[j][i]
                for k in range(i, 2 * n):
                    augmented[j][k] -= factor * augmented[i][k]
    
    # Extract the inverse matrix
    inverse = [row[n:] for row in augmented]
    
    return inverse