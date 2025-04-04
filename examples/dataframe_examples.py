#!/usr/bin/env python3
"""
DataFrame operation examples for pyroid.

This script demonstrates the DataFrame capabilities of pyroid.
"""

import time
import random
import pandas as pd
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def main():
    print("pyroid DataFrame Operations Examples")
    print("===================================")
    
    # Example 1: DataFrame Apply
    print("\n1. DataFrame Apply")
    
    # Create a test DataFrame
    n_rows = 1_000_000
    df = {
        'A': [random.random() for _ in range(n_rows)],
        'B': [random.random() for _ in range(n_rows)],
        'C': [random.random() for _ in range(n_rows)]
    }
    
    # Convert to pandas DataFrame for comparison
    pandas_df = pd.DataFrame(df)
    
    print("\nApplying a function to each column:")
    
    # Define a function to apply
    def square(x):
        return [val * val for val in x]
    
    print("\nPandas apply:")
    pandas_result = benchmark(lambda: pandas_df.apply(lambda x: x ** 2))
    
    print("\npyroid parallel apply:")
    pyroid_result = benchmark(lambda: pyroid.dataframe_apply(df, square, 0))
    
    print("\nResults (first 5 rows):")
    print(f"Pandas:\n{pandas_result.head()}")
    print(f"pyroid: {dict((k, v[:5]) for k, v in pyroid_result.items())}")
    
    # Example 2: GroupBy Aggregate
    print("\n2. GroupBy Aggregate")
    
    # Create a test DataFrame with groups
    n_rows = 1_000_000
    categories = ['A', 'B', 'C', 'D', 'E']
    df = {
        'category': [random.choice(categories) for _ in range(n_rows)],
        'value1': [random.random() * 100 for _ in range(n_rows)],
        'value2': [random.random() * 100 for _ in range(n_rows)]
    }
    
    # Convert to pandas DataFrame for comparison
    pandas_df = pd.DataFrame(df)
    
    print("\nGrouping by category and calculating aggregates:")
    
    print("\nPandas groupby:")
    pandas_result = benchmark(lambda: pandas_df.groupby('category').agg({
        'value1': 'mean',
        'value2': 'sum'
    }))
    
    print("\npyroid parallel groupby:")
    agg_dict = {'value1': 'mean', 'value2': 'sum'}
    pyroid_result = benchmark(lambda: pyroid.dataframe_groupby_aggregate(df, ['category'], agg_dict))
    
    print("\nResults:")
    print(f"Pandas:\n{pandas_result}")
    print(f"pyroid: {pyroid_result}")
    
    # Example 3: Transform
    print("\n3. Parallel Transform")
    
    # Create a test DataFrame
    n_rows = 1_000_000
    df = {
        'A': [random.random() * 100 for _ in range(n_rows)],
        'B': [random.random() * 100 for _ in range(n_rows)],
        'C': [random.random() * 100 for _ in range(n_rows)]
    }
    
    # Convert to pandas DataFrame for comparison
    pandas_df = pd.DataFrame(df)
    
    print("\nApplying multiple transformations:")
    
    print("\nPandas transform:")
    pandas_result = benchmark(lambda: pandas_df.assign(
        A_log=lambda x: x['A'].apply(lambda y: y if y <= 0 else np.log(y)),
        B_sqrt=lambda x: x['B'].apply(lambda y: y ** 0.5),
        C_round=lambda x: x['C'].round(2)
    ))
    
    print("\npyroid parallel transform:")
    transformations = [
        ('A', 'log', None),
        ('B', 'sqrt', None),
        ('C', 'round', 2)
    ]
    pyroid_result = benchmark(lambda: pyroid.parallel_transform(df, transformations))
    
    print("\nResults (first 5 rows):")
    print(f"Pandas:\n{pandas_result.head()}")
    print(f"pyroid: {dict((k, v[:5]) for k, v in pyroid_result.items())}")
    
    # Example 4: Join
    print("\n4. Parallel Join")
    
    # Create test DataFrames
    n_rows = 500_000
    left_df = {
        'id': list(range(n_rows)),
        'value_left': [random.random() * 100 for _ in range(n_rows)]
    }
    
    n_rows_right = 300_000
    right_df = {
        'id': [random.randint(0, n_rows-1) for _ in range(n_rows_right)],
        'value_right': [random.random() * 100 for _ in range(n_rows_right)]
    }
    
    # Convert to pandas DataFrames for comparison
    pandas_left = pd.DataFrame(left_df)
    pandas_right = pd.DataFrame(right_df)
    
    print("\nJoining two DataFrames:")
    
    print("\nPandas join:")
    pandas_result = benchmark(lambda: pandas_left.merge(pandas_right, on='id', how='inner'))
    
    print("\npyroid parallel join:")
    pyroid_result = benchmark(lambda: pyroid.parallel_join(left_df, right_df, 'id', 'inner'))
    
    print("\nResults (shape):")
    print(f"Pandas: {pandas_result.shape}")
    print(f"pyroid: ({len(next(iter(pyroid_result.values())))}, {len(pyroid_result)})")

if __name__ == "__main__":
    main()