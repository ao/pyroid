# DataFrame Operations

This document provides detailed information about the DataFrame operations available in Pyroid.

## DataFrame Class

The `DataFrame` class provides a way to store and manipulate tabular data.

### Constructor

```python
pyroid.data.DataFrame(data)
```

Creates a new DataFrame from a dictionary of columns.

**Parameters:**
- `data` (dict): A dictionary where keys are column names and values are lists of column values.

**Returns:**
- A new DataFrame object.

**Example:**
```python
import pyroid

# Create a DataFrame
df = pyroid.data.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "age": [25, 30, 35, 40, 45]
})
```

## Apply Function

```python
pyroid.data.apply(df, func, axis=0)
```

Applies a function to each column or row of a DataFrame.

**Parameters:**
- `df` (DataFrame): The DataFrame to apply the function to.
- `func` (callable): The function to apply.
- `axis` (int, optional): The axis along which to apply the function. 0 for columns, 1 for rows. Default is 0.

**Returns:**
- A new DataFrame with the function applied.

**Example:**
```python
import pyroid

# Create a DataFrame
df = pyroid.data.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "age": [25, 30, 35, 40, 45]
})

# Apply a function to each column
result = pyroid.data.apply(df, lambda x: x * 2, axis=0)
```

## GroupBy Aggregate

```python
pyroid.data.groupby_aggregate(df, by, agg_dict)
```

Groups a DataFrame by one or more columns and applies aggregation functions.

**Parameters:**
- `df` (DataFrame): The DataFrame to group.
- `by` (str or list): The column(s) to group by.
- `agg_dict` (dict): A dictionary mapping column names to aggregation functions.

**Returns:**
- A new DataFrame with the grouped and aggregated data.

**Example:**
```python
import pyroid

# Create a DataFrame
df = pyroid.data.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "age": [25, 30, 25, 30, 35]
})

# Group by age and count names
grouped = pyroid.data.groupby_aggregate(df, "age", {"name": "count"})
```

## Filter

```python
pyroid.data.filter(df, condition)
```

Filters a DataFrame based on a condition.

**Parameters:**
- `df` (DataFrame): The DataFrame to filter.
- `condition` (callable): A function that takes a row and returns a boolean.

**Returns:**
- A new DataFrame with only the rows that satisfy the condition.

**Example:**
```python
import pyroid

# Create a DataFrame
df = pyroid.data.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "age": [25, 30, 35, 40, 45]
})

# Filter rows where age > 30
filtered = pyroid.data.filter(df, lambda row: row["age"] > 30)
```

## Sort

```python
pyroid.data.sort(df, by, ascending=True)
```

Sorts a DataFrame by one or more columns.

**Parameters:**
- `df` (DataFrame): The DataFrame to sort.
- `by` (str or list): The column(s) to sort by.
- `ascending` (bool or list, optional): Whether to sort in ascending order. Default is True.

**Returns:**
- A new DataFrame sorted by the specified columns.

**Example:**
```python
import pyroid

# Create a DataFrame
df = pyroid.data.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "age": [25, 30, 35, 40, 45]
})

# Sort by age in descending order
sorted_df = pyroid.data.sort(df, "age", ascending=False)
```

## Join

```python
pyroid.data.join(left_df, right_df, on, how="inner")
```

Joins two DataFrames on a common column.

**Parameters:**
- `left_df` (DataFrame): The left DataFrame.
- `right_df` (DataFrame): The right DataFrame.
- `on` (str): The column to join on.
- `how` (str, optional): The type of join to perform. One of "inner", "left", "right", "outer". Default is "inner".

**Returns:**
- A new DataFrame with the joined data.

**Example:**
```python
import pyroid

# Create two DataFrames
employees = pyroid.data.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "department_id": [1, 2, 1, 3, 2]
})

departments = pyroid.data.DataFrame({
    "id": [1, 2, 3],
    "name": ["HR", "Engineering", "Marketing"]
})

# Join employees and departments
joined = pyroid.data.join(employees, departments, on="id", how="inner")
```

## Pivot

```python
pyroid.data.pivot(df, index, columns, values)
```

Creates a pivot table from a DataFrame.

**Parameters:**
- `df` (DataFrame): The DataFrame to pivot.
- `index` (str): The column to use as the index.
- `columns` (str): The column to use as the columns.
- `values` (str): The column to use as the values.

**Returns:**
- A new DataFrame with the pivoted data.

**Example:**
```python
import pyroid

# Create a DataFrame
df = pyroid.data.DataFrame({
    "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
    "product": ["A", "B", "A", "B"],
    "sales": [100, 200, 150, 250]
})

# Create a pivot table
pivot = pyroid.data.pivot(df, index="date", columns="product", values="sales")
```

## Melt

```python
pyroid.data.melt(df, id_vars, value_vars, var_name="variable", value_name="value")
```

Unpivots a DataFrame from wide to long format.

**Parameters:**
- `df` (DataFrame): The DataFrame to melt.
- `id_vars` (list): Columns to use as identifier variables.
- `value_vars` (list): Columns to unpivot.
- `var_name` (str, optional): Name to use for the variable column. Default is "variable".
- `value_name` (str, optional): Name to use for the value column. Default is "value".

**Returns:**
- A new DataFrame in long format.

**Example:**
```python
import pyroid

# Create a DataFrame
df = pyroid.data.DataFrame({
    "date": ["2023-01-01", "2023-01-02"],
    "product_A": [100, 150],
    "product_B": [200, 250]
})

# Melt the DataFrame
melted = pyroid.data.melt(df, id_vars=["date"], value_vars=["product_A", "product_B"], var_name="product", value_name="sales")