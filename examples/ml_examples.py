#!/usr/bin/env python3
"""
Machine learning operation examples for pyroid.

This script demonstrates the machine learning capabilities of pyroid.
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
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
    
    # Example 1: K-means Clustering
    print("\n1. K-means Clustering")
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    
    # Create three clusters
    n_samples = 300
    centers = [(0, 0), (5, 5), (0, 5)]
    cluster_std = [0.8, 1.0, 0.5]
    
    X = []
    for i, (cx, cy) in enumerate(centers):
        cluster_points = n_samples // 3
        for _ in range(cluster_points):
            x = np.random.normal(cx, cluster_std[i])
            y = np.random.normal(cy, cluster_std[i])
            X.append([x, y])
    
    print(f"Generated {len(X)} points in 3 clusters")
    
    # Run K-means clustering
    print("\nRunning K-means clustering with k=3:")
    result = benchmark(lambda: pyroid.ml.basic.kmeans(X, k=3))
    
    # Extract results
    centroids = result['centroids']
    clusters = result['clusters']
    iterations = result['iterations']
    
    print(f"K-means converged in {iterations} iterations")
    print(f"Centroids: {centroids}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    X_np = np.array(X)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=clusters, cmap='viridis')
    centroids_np = np.array(centroids)
    plt.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='X', s=100)
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('kmeans_clustering.png')
    print("Plot saved as 'kmeans_clustering.png'")
    
    # Example 2: Linear Regression
    print("\n2. Linear Regression")
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    
    # Create linear data with noise
    n_samples = 100
    x = list(range(1, n_samples + 1))
    y = [2 * xi + 5 + np.random.normal(0, 10) for xi in x]
    
    print(f"Generated {len(x)} data points")
    
    # Run linear regression
    print("\nRunning linear regression:")
    result = benchmark(lambda: pyroid.ml.basic.linear_regression(x, y))
    
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
    plt.savefig('linear_regression.png')
    print("Plot saved as 'linear_regression.png'")
    
    # Example 3: Data Normalization
    print("\n3. Data Normalization")
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    
    # Create data with different scales
    values = [np.random.normal(50, 10) for _ in range(100)]
    
    print(f"Generated {len(values)} values")
    print(f"Original data - Min: {min(values):.2f}, Max: {max(values):.2f}, Mean: {sum(values)/len(values):.2f}")
    
    # Run normalization
    print("\nRunning min-max normalization:")
    minmax_normalized = benchmark(lambda: pyroid.ml.basic.normalize(values, method='minmax'))
    
    print("\nRunning z-score normalization:")
    zscore_normalized = benchmark(lambda: pyroid.ml.basic.normalize(values, method='zscore'))
    
    # Print statistics
    print(f"\nMin-Max normalized - Min: {min(minmax_normalized):.2f}, Max: {max(minmax_normalized):.2f}")
    print(f"Z-Score normalized - Mean: {sum(zscore_normalized)/len(zscore_normalized):.2f}, Std: {np.std(zscore_normalized):.2f}")
    
    # Plot the results
    plt.figure(figsize=(15, 5))
    
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
    plt.savefig('normalization.png')
    print("Plot saved as 'normalization.png'")
    
    # Example 4: Distance Matrix
    print("\n4. Distance Matrix")
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    
    # Create points in 2D space
    points = [[np.random.random() * 10, np.random.random() * 10] for _ in range(5)]
    
    print(f"Generated {len(points)} points")
    
    # Run distance matrix calculation
    print("\nCalculating Euclidean distance matrix:")
    euclidean_distances = benchmark(lambda: pyroid.ml.basic.distance_matrix(points, metric='euclidean'))
    
    print("\nCalculating Manhattan distance matrix:")
    manhattan_distances = benchmark(lambda: pyroid.ml.basic.distance_matrix(points, metric='manhattan'))
    
    # Print results
    print("\nEuclidean distance matrix:")
    for row in euclidean_distances:
        print([f"{val:.2f}" for val in row])
    
    print("\nManhattan distance matrix:")
    for row in manhattan_distances:
        print([f"{val:.2f}" for val in row])
    
    # Plot the results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Euclidean distances
    im1 = axes[0].imshow(euclidean_distances, cmap='viridis')
    axes[0].set_title('Euclidean Distance Matrix')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot Manhattan distances
    im2 = axes[1].imshow(manhattan_distances, cmap='viridis')
    axes[1].set_title('Manhattan Distance Matrix')
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('distance_matrices.png')
    print("Plot saved as 'distance_matrices.png'")

if __name__ == "__main__":
    main()