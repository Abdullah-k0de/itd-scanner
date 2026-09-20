# itd-scanner

[![PyPI version](https://img.shields.io/pypi/v/itd-scanner.svg)](https://pypi.org/project/itd-scanner/0.1.0/)
[![Python Version](https://img.shields.io/pypi/pyversions/itd-scanner.svg)](https://pypi.org/project/itd-scanner/0.1.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-green.svg)](https://peps.python.org/pep-0008/)

> 🚀 **Official PyPI Release:** [https://pypi.org/project/itd-scanner/0.1.0/](https://pypi.org/project/itd-scanner/0.1.0/)

A production-ready static analysis engine for detecting **Idiomatic Technical Debt (ITD)**, vectorization bypass, and performance antipatterns in Data Science Python (**Pandas, NumPy, SciPy, and PyTorch**).


---

## What is Idiomatic Technical Debt (ITD)?

Modern LLMs and procedural programmers often generate functionally correct data science code that bypasses high-performance vectorized C/Fortran primitives. Instead, they produce slow, memory-intensive Python loops, accumulator variables, and reinvented algorithms.

Traditional linters (`pylint`, `flake8`, `perflint`) focus on PEP 8 style or general Python idioms and miss vectorized data science smells. **`itd-scanner`** inspects the Abstract Syntax Tree (AST) to pinpoint exactly where vectorized libraries are being bypassed.

---

## The 5 Core ITD Antipatterns

| Smell                       | Antipattern (What It Detects)                                                                                          | Idiomatic Alternative                                               |
| :-------------------------- | :--------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------ |
| **Collection Choker** | Row-by-row iteration over DataFrames/Arrays (`.iterrows()`, `.itertuples()`, `for i in range(len(arr)): arr[i]`) | Vectorized operations,`.apply()`, or array indexing               |
| **Math Looper**       | Scalar accumulator math (`total += arr[i]` or `total += x * y`) inside loops                                       | `np.sum()`, `df['col'].sum()`, or vectorized ufuncs             |
| **String Masher**     | Quadratic string concatenation (`s += ...`) inside iterative loops                                                   | `str.join()`, `f-strings`, or `pd.Series.str`                 |
| **RAM Hog**           | Wasteful list comprehensions materialized inside reductions (`sum([...])`, `set([...])`)                           | Generator expressions`sum(...)` or set comprehensions `{...}`   |
| **DIY Wheel**         | Reinventing built-ins (`d[k] = d.get(k, 0) + 1`, linear search loops, manual `min`/`max`/`unique`)             | `collections.Counter()`, `pd.value_counts()`, `in`, `set()` |

---

## Quickstart & CLI

### Installation

```bash
# Install core package from PyPI
pip install itd-scanner

# Or install with research baseline linters (dslinter, perflint, pandas-vet)
pip install "itd-scanner[baselines]"
```

### CLI Usage

Scan any Python file directly from your terminal:

```bash
# Scan for ITD smells with ANSI colored reporting
itd-scanner my_script.py

# Include Radon software metrics (Cyclomatic Complexity, SLOC, Halstead Volume, MI)
itd-scanner my_script.py --metrics

# Output structured JSON for CI/CD pipelines
itd-scanner my_script.py --json
```

#### Terminal Preview

```text
=================================================================
  ITD-Scanner Report: analysis.py
=================================================================

Idiomatic Technical Debt (ITD) Smells:
  Collection Choker     : [DETECTED]
  Math Looper           : [CLEAN]
  String Masher         : [CLEAN]
  RAM Hog               : [DETECTED]
  DIY Wheel             : [CLEAN]

Detailed Findings (2):
  Line 14  [Collection Choker]: Row-by-row DataFrame iteration using .iterrows() instead of vectorized operations.
  Line 28  [RAM Hog]: Wasteful list comprehension materialized inside 'sum()'; use a generator expression or vectorized call.

Software Complexity & Maintainability (Radon):
  Lines of Code (LOC)      : 34
  Source Lines (SLOC)      : 22
  Cyclomatic Complexity (CC): 4
  Maintainability Index    : 84.12/100
  Halstead Volume          : 312.45 bits
  Halstead Difficulty      : 8.50

=================================================================
 Status: 2 ITD Smells Found
=================================================================
```

---

## Python API

You can easily embed `itd-scanner` into custom analysis pipelines, linters, or research benchmarks:

```python
from itd_scanner import scan_code, scan_file

code = """
import pandas as pd

total = 0
for idx, row in df.iterrows():
    total += row['price']
"""

# Simple boolean smell flags
flags = scan_code(code)
print(flags)
# Output:
# {
#   'Collection Choker': True,
#   'Math Looper': True,
#   'String Masher': False,
#   'RAM Hog': False,
#   'DIY Wheel': False
# }

# Detailed report with AST line numbers and Radon software metrics
report = scan_code(code, return_details=True, include_metrics=True)
print(report["findings"])
# [
#   {'smell': 'Collection Choker', 'line': 5, 'message': 'Row-by-row DataFrame iteration...'},
#   {'smell': 'Math Looper', 'line': 6, 'message': 'Accumulator math inside loop...'}
# ]
print(report["metrics"]["cyclomatic_complexity"])  # CC
```

---

## Antipattern Examples & Fixes

### 1. Collection Choker

```python
# ❌ Antipattern (Iterating row-by-row)
for idx, row in df.iterrows():
    df.loc[idx, 'total'] = row['a'] + row['b']

# ✅ Idiomatic (Vectorized column addition)
df['total'] = df['a'] + df['b']
```

### 2. Math Looper

```python
# ❌ Antipattern (Pure Python loop math)
total = 0
for i in range(len(arr)):
    total += arr[i] * 2

# ✅ Idiomatic (NumPy vectorized ufunc)
total = np.sum(arr * 2)
```

### 3. DIY Wheel (Frequency Counting)

```python
# ❌ Antipattern (Manual dict accumulation)
counts = {}
for item in items:
    counts[item] = counts.get(item, 0) + 1

# ✅ Idiomatic (Standard library Counter / Pandas)
from collections import Counter
counts = Counter(items)
```

---

## Research & Empirical Study

This library is part of an academic research project at King Fahd University of Petroleum & Minerals (KFUPM) analyzing code generated by Large Language Models across the **DS-1000** benchmark.

### Citation

If you use `itd-scanner` in academic research, please cite:

```bibtex
@misc{khalid2026itdscanner,
  author = {Abdullah Khalid},
  title = {itd-scanner: Static Analysis Engine for Idiomatic Technical Debt in Data Science Python},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Abdullah-k0de/itd-scanner}}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
