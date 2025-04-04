# Pyroid Documentation Guides

Welcome to the Pyroid documentation guides. These guides provide comprehensive information on how to use Pyroid effectively, from getting started to advanced usage patterns and performance optimization.

## Available Guides

| Guide | Description |
|-------|-------------|
| [Getting Started](./getting_started.md) | Introduction to Pyroid and basic usage examples |
| [Performance Optimization](./performance.md) | Tips and best practices for optimizing performance |
| [Advanced Usage](./advanced_usage.md) | Advanced usage patterns and techniques |
| [Migration Guide](./migration.md) | Guide for migrating from other libraries to Pyroid |
| [FAQ](./faq.md) | Frequently asked questions about Pyroid |

## Guide Selection

Not sure which guide to read? Here's a quick reference:

- **New to Pyroid?** Start with the [Getting Started](./getting_started.md) guide.
- **Want to improve performance?** Check out the [Performance Optimization](./performance.md) guide.
- **Looking for advanced techniques?** Read the [Advanced Usage](./advanced_usage.md) guide.
- **Migrating from another library?** Refer to the [Migration Guide](./migration.md).
- **Have specific questions?** See if they're answered in the [FAQ](./faq.md).

## Core Concepts

Before diving into the specific guides, it's helpful to understand some core concepts in Pyroid:

### Parallel Processing

Pyroid achieves its performance advantages primarily through parallel processing. Most functions in Pyroid are designed to process multiple items in parallel, utilizing all available CPU cores.

```python
# Example of parallel processing
import pyroid

# Process multiple items in parallel
data = [1, 2, 3, 4, 5]
squared = pyroid.parallel_map(data, lambda x: x * x)
```

### Rust Implementation

Under the hood, Pyroid's functions are implemented in Rust, a systems programming language that offers performance comparable to C/C++ with memory safety guarantees. The Python interface is generated using PyO3, a Rust library that provides bindings between Rust and Python.

### Python Integration

Despite being implemented in Rust, Pyroid is designed to integrate seamlessly with Python code. It accepts Python-native data structures as input and returns Python-native data structures as output.

```python
# Example of Python integration
import pyroid
import pandas as pd

# Use Pyroid with pandas
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = pyroid.dataframe_apply(df.to_dict('list'), lambda x: [val * 2 for val in x], 0)
df_result = pd.DataFrame(result)
```

### Domain-Specific Modules

Pyroid is organized into domain-specific modules, each focusing on a specific area such as math operations, string processing, data manipulation, etc. This modular design makes it easy to find the functions you need for your specific use case.

## Learning Path

If you're new to Pyroid, we recommend following this learning path:

1. **Getting Started**: Learn the basics of Pyroid and how to use its core functions.
2. **API Reference**: Explore the [API reference](../api/index.md) to understand the available functions and their parameters.
3. **Performance Optimization**: Learn how to optimize your code for maximum performance.
4. **Advanced Usage**: Discover advanced techniques and patterns for using Pyroid effectively.

## Additional Resources

In addition to these guides, you might find these resources helpful:

- **[API Reference](../api/index.md)**: Detailed documentation of all Pyroid functions and classes.
- **[Examples](../../examples/)**: Example code demonstrating various Pyroid features.
- **[Benchmarks](../../benchmarks/)**: Performance benchmarks comparing Pyroid to other libraries.
- **[GitHub Repository](https://github.com/ao/pyroid)**: Source code and issue tracker.

## Contributing to Documentation

We welcome contributions to the Pyroid documentation! If you find any errors or have suggestions for improvement, please submit an issue or pull request on the [GitHub repository](https://github.com/ao/pyroid).

When contributing to documentation, please follow these guidelines:

1. **Be clear and concise**: Use simple language and avoid jargon.
2. **Include examples**: Provide code examples to illustrate concepts.
3. **Follow the style**: Match the existing documentation style.
4. **Test code examples**: Ensure that all code examples work as expected.

## Getting Help

If you need help with Pyroid, you can:

- Check the [FAQ](./faq.md) for answers to common questions.
- Search for similar issues on the [GitHub issue tracker](https://github.com/ao/pyroid/issues).
- Ask a question on [Stack Overflow](https://stackoverflow.com/questions/tagged/pyroid) with the `pyroid` tag.
