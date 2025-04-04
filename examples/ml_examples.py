#!/usr/bin/env python3
"""
Machine learning operation examples for pyroid.

This script demonstrates the machine learning capabilities of pyroid.
"""

import time
import random
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def main():
    print("pyroid Machine Learning Operations Examples")
    print("=========================================")
    
    # Example 1: Distance Matrix
    print("\n1. Distance Matrix")
    
    # Generate random points
    n_points = 2000
    points = [[random.random() for _ in range(10)] for _ in range(n_points)]
    points_np = np.array(points)
    
    print(f"\nCalculating distance matrix for {n_points} points with 10 dimensions:")
    
    print("\nNumPy distance matrix:")
    def numpy_distance_matrix(points):
        n = len(points)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix
    
    numpy_result = benchmark(lambda: numpy_distance_matrix(points_np))
    
    print("\npyroid parallel distance matrix:")
    pyroid_result = benchmark(lambda: pyroid.parallel_distance_matrix(points, "euclidean"))
    
    print("\nResults (shape):")
    print(f"NumPy: {numpy_result.shape}")
    print(f"pyroid: ({len(pyroid_result)}, {len(pyroid_result[0])})")
    
    # Example 2: Feature Scaling
    print("\n2. Feature Scaling")
    
    # Generate random data
    n_samples = 100000
    n_features = 20
    data = [[random.gauss(0, 10) for _ in range(n_features)] for _ in range(n_samples)]
    data_np = np.array(data)
    
    print(f"\nScaling {n_samples} samples with {n_features} features:")
    
    print("\nScikit-learn StandardScaler:")
    scaler = StandardScaler()
    sklearn_result = benchmark(lambda: scaler.fit_transform(data_np))
    
    print("\npyroid parallel feature scaling:")
    pyroid_result = benchmark(lambda: pyroid.parallel_feature_scaling(data, "standard"))
    
    print("\nResults (shape):")
    print(f"Scikit-learn: {sklearn_result.shape}")
    print(f"pyroid: ({len(pyroid_result)}, {len(pyroid_result[0])})")
    
    # Example 3: Cross Validation
    print("\n3. Cross Validation")
    
    # Generate a classification dataset
    X, y = make_blobs(n_samples=10000, centers=3, n_features=10, random_state=42)
    X = X.tolist()
    y = y.tolist()
    
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
    
    print("\nPerforming 5-fold cross-validation:")
    
    print("\nScikit-learn cross_val_score:")
    model = KNeighborsClassifier(n_neighbors=3)
    sklearn_result = benchmark(lambda: cross_val_score(model, np.array(X), np.array(y), cv=5))
    
    print("\npyroid parallel cross-validation:")
    pyroid_result = benchmark(lambda: pyroid.parallel_cross_validation(X, y, 5, knn_model, accuracy))
    
    print("\nResults (scores):")
    print(f"Scikit-learn: {sklearn_result}")
    print(f"pyroid: {pyroid_result}")

if __name__ == "__main__":
    main()