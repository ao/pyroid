import nbformat as nbf
import json

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add markdown cell with title
nb.cells.append(nbf.v4.new_markdown_cell("# Pyroid Performance Benchmarks\n\nThis notebook demonstrates the performance advantages of Pyroid compared to pure Python implementations."))

# Add imports and setup
imports = """import time
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Try to import pyroid
try:
    import pyroid
    PYROID_AVAILABLE = True
except ImportError:
    print("Warning: pyroid not found. Please install pyroid to run benchmarks.")
    PYROID_AVAILABLE = False"""
nb.cells.append(nbf.v4.new_code_cell(imports))

# Write the notebook to a file
with open('pyroid_benchmarks.ipynb', 'w') as f:
    json.dump(nb, f)

print("Basic notebook created. Run this script to add more sections.")
