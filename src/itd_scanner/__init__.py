"""
itd-scanner: Static Analysis for Idiomatic Technical Debt in Data Science Python.
"""

import ast
from typing import Dict, Any, Optional
from itd_scanner.visitor import ITDScanner
from itd_scanner.metrics import compute_code_metrics
from itd_scanner.baselines import run_all_baselines, run_dslinter, run_perflint, run_pandas_vet

__version__ = "0.1.0"

def scan_code(source_code: str, return_details: bool = False, include_metrics: bool = False) -> Dict[str, Any]:
    """
    Scans a Python code snippet for the 5 Idiomatic Technical Debt (ITD) anti-patterns,
    with optional software complexity metrics (Radon CC, LOC).

    Args:
        source_code (str): The Python code to analyze.
        return_details (bool): If True, returns detailed finding messages and line numbers.
        include_metrics (bool): If True, also computes Radon complexity and LOC metrics.

    Returns:
        dict: Detection flags and optional metrics/findings.
    """
    scanner = ITDScanner()
    syntax_error = False

    try:
        tree = ast.parse(source_code)
        scanner.visit(tree)
    except (SyntaxError, ValueError):
        syntax_error = True

    output: Dict[str, Any] = {}

    if return_details:
        output["flags"] = scanner.flags
        output["findings"] = scanner.findings
        output["syntax_error"] = syntax_error
    else:
        output.update(scanner.flags)

    if include_metrics:
        output["metrics"] = compute_code_metrics(source_code)

    return output

def scan_file(file_path: str, return_details: bool = False, include_metrics: bool = False) -> Dict[str, Any]:
    """
    Reads a Python file and scans it for ITD anti-patterns.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    return scan_code(code, return_details=return_details, include_metrics=include_metrics)

__all__ = [
    "ITDScanner",
    "scan_code",
    "scan_file",
    "compute_code_metrics",
    "run_all_baselines",
    "run_dslinter",
    "run_perflint",
    "run_pandas_vet",
]
