# itd-scanner

A static analysis engine for detecting **Idiomatic Technical Debt (ITD)** and vectorized library bypass in Data Science Python (Pandas, NumPy, SciPy, PyTorch).

## Features

- **AST ITD Visitor**: Scans Python source code to detect 5 primary anti-patterns:
  1. **Collection Choker**: Non-vectorized iteration (`.iterrows()`, `.itertuples()`, manual DataFrame/Series loops).
  2. **Math Looper**: Pure Python accumulator loops bypassing NumPy/SciPy ufuncs.
  3. **String Masher**: Inefficient `+=` string concatenation inside iterative loops.
  4. **RAM Hog**: Materializing intermediate list comprehensions inside aggregation functions (`sum()`, `any()`, `all()`).
  5. **DIY Wheel**: Reinventing standard algorithms (grouping, counting, searching) using raw loops.
- **Code Metrics**: Integrated Cyclomatic Complexity (CC) and raw Lines of Code (LOC) via `radon`.
- **Baseline Benchmarking**: Subprocess runners for external linters (`dslinter`, `perflint`, `pandas-vet`).

## Installation

```bash
# Core package
pip install -e .

# With baseline benchmarking linters (for thesis experiments)
pip install -e ".[baselines]"
```

## Usage

```python
from itd_scanner import scan_code

code = """
import pandas as pd
for index, row in df.iterrows():
    total += row['value']
"""

results = scan_code(code)
print(results)
```
