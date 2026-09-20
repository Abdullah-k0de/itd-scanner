"""
itd-scanner: Static Analysis for Idiomatic Technical Debt in Data Science Python.
"""

import ast
from typing import Dict, Any, Union
from itd_scanner.visitor import ITDScanner

__version__ = "0.1.0"

def scan_code(source_code: str, return_details: bool = False) -> Dict[str, Any]:
    """
    Scans a Python code snippet for the 5 Idiomatic Technical Debt (ITD) anti-patterns.

    Args:
        source_code (str): The Python code to analyze.
        return_details (bool): If True, returns a dictionary with flags, findings list,
                               and syntax error status. If False, returns boolean flags only.

    Returns:
        dict: Detection flags and optional findings.
    """
    scanner = ITDScanner()
    syntax_error = False

    try:
        tree = ast.parse(source_code)
        scanner.visit(tree)
    except (SyntaxError, ValueError) as e:
        syntax_error = True

    if return_details:
        return {
            "flags": scanner.flags,
            "findings": scanner.findings,
            "syntax_error": syntax_error,
        }

    return scanner.flags

def scan_file(file_path: str, return_details: bool = False) -> Dict[str, Any]:
    """
    Reads a Python file and scans it for ITD anti-patterns.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    return scan_code(code, return_details=return_details)

__all__ = ["ITDScanner", "scan_code", "scan_file"]
