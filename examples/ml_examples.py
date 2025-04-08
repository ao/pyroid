#!/usr/bin/env python3
"""
Machine learning operation examples for pyroid.

This script demonstrates the machine learning capabilities of pyroid.
"""

import time
import random
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def main():
    print("Pyroid Machine Learning Examples")
    print("=============================")
    
    # Example 1: K-means clustering
    print("\n1. K-means Clustering")
    
    # Generate sample data
    print("\nGenerating sample data...")
    random.seed(42)  # For reproducibility
    
    # Create three clusters
    data = []
    
    # Cluster 1: around (1, 2)
    for _ in range(20):
        x = 1 + random.random()
        y = 2 + random.random()
        data.append([x, y])
    
    # Cluster 2: around (5, 7)
    for _ in range(20):
        x = 5 + random.random()
        y = 7 + random.random()
        data.append([x, y])
    
    # Cluster 3: around (9, 3)
    for _ in range(20):
        x = 9 + random.random()
        y = 3 + random.random()
        data.append([x, y])
    
    print(f"Generated {len(data)} data points")
    
    # Run K-means clustering
    print("\nRunning K-means clustering with k=3:")
    result = benchmark(lambda: pyroid.ml.basic.kmeans(data, k=3))
    
    # Extract results
    centroids = result['centroids']
    clusters = result['clusters']
    iterations = result.get('iterations', 'N/A')
    
    print(f"K-means converged in {iterations} iterations")
    print(f"Centroids: {centroids}")
    print(f"Cluster distribution: {[clusters.count(i) for i in range(3)]}")
    
    # Example 2: Linear regression
    print("\n2. Linear Regression")
    
    # Generate sample data
    print("\nGenerating sample data...")
    random.seed(42)  # For reproducibility
    
    # Create linear data with noise
    X = []
    y = []
    
    for i in range(100):
        x1 = random.random() * 10
        x2 = random.random() * 5
        # y = 2*x1 + 3*x2 + 5 + noise
        y_val = 2 * x1 + 3 * x2 + 5 + random.random() * 2 - 1
        X.append([x1, x2])
        y.append(y_val)
    
    print(f"Generated {len(X)} data points")
    
    # Run linear regression
    print("\nRunning linear regression:")
    result = benchmark(lambda: pyroid.ml.basic.linear_regression(X, y))
    
    # Extract results
    coefficients = result.get('coefficients', [])
    intercept = result.get('intercept', 0)
    r_squared = result.get('r_squared', 0)
    
    print(f"Coefficients: {coefficients}")
    print(f"Intercept: {intercept}")
    print(f"R-squared: {r_squared}")
    print(f"True model: y = 2*x1 + 3*x2 + 5")
    print(f"Fitted model: y = {coefficients[0]:.2f}*x1 + {coefficients[1]:.2f}*x2 + {intercept:.2f}")
    
    # Example 3: Data normalization
    print("\n3. Data Normalization")
    
    # Generate sample data
    print("\nGenerating sample data...")
    random.seed(42)  # For reproducibility
    
    # Create data with different scales
    values = [random.random() * 100 for _ in range(100)]
    
    print(f"Generated {len(values)} values")
    print(f"Original data - Min: {min(values):.2f}, Max: {max(values):.2f}, Mean: {sum(values)/len(values):.2f}")
    
    # Run normalization
    print("\nRunning min-max normalization:")
    minmax_normalized = benchmark(lambda: pyroid.ml.basic.normalize(values, method="min-max"))
    
    print("\nRunning z-score normalization:")
    try:
        zscore_normalized = benchmark(lambda: pyroid.ml.basic.normalize(values, method="z-score"))
    except Exception as e:
        print(f"Z-score normalization not implemented: {e}")
        zscore_normalized = []
    
    # Print statistics
    print(f"\nMin-Max normalized - Min: {min(minmax_normalized):.2f}, Max: {max(minmax_normalized):.2f}")
    if zscore_normalized:
        print(f"Z-Score normalized - Mean: {sum(zscore_normalized)/len(zscore_normalized):.2f}, Std: {(sum((x - sum(zscore_normalized)/len(zscore_normalized))**2 for x in zscore_normalized)/len(zscore_normalized))**0.5:.2f}")
    
    # Example 4: Distance matrix
    print("\n4. Distance Matrix")
    
    # Generate sample data
    print("\nGenerating sample data...")
    random.seed(42)  # For reproducibility
    
    # Create points in 2D space
    points = [[random.random() * 10, random.random() * 10] for _ in range(5)]
    
    print(f"Generated {len(points)} points")
    for i, point in enumerate(points):
        print(f"Point {i}: ({point[0]:.2f}, {point[1]:.2f})")
    
    # Run distance matrix calculation
    print("\nCalculating Euclidean distance matrix:")
    euclidean_distances = benchmark(lambda: pyroid.ml.basic.distance_matrix(points, metric="euclidean"))
    
    print("\nCalculating Manhattan distance matrix:")
    try:
        manhattan_distances = benchmark(lambda: pyroid.ml.basic.distance_matrix(points, metric="manhattan"))
    except Exception as e:
        print(f"Manhattan distance not implemented: {e}")
        manhattan_distances = []
    
    # Print results
    print("\nEuclidean distance matrix:")
    for row in euclidean_distances:
        print([f"{val:.2f}" for val in row])
    
    if manhattan_distances:
        print("\nManhattan distance matrix:")
        for row in manhattan_distances:
            print([f"{val:.2f}" for val in row])
    
    print("\nMachine learning examples completed.")

if __name__ == "__main__":
    main()