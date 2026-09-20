"""
Code complexity and structural metrics using Radon.
Computes Cyclomatic Complexity (CC), Raw Metrics (LOC, LLOC, SLOC), and Maintainability Index (MI).
"""

import ast
from typing import Dict, Any
from radon.complexity import cc_visit
from radon.raw import analyze
from radon.metrics import mi_visit, h_visit

def _wrap_in_function(code: str) -> str:
    """
    Wraps top-level code inside a synthetic function definition.
    This ensures radon's cc_visit calculates Cyclomatic Complexity for top-level loops and branches.
    """
    lines = code.splitlines()
    indented = "\n".join("    " + line for line in lines)
    return f"def __snippet_scope__():\n{indented}\n"

def compute_code_metrics(source_code: str) -> Dict[str, Any]:
    """
    Calculates software metrics for a given Python code snippet:
    - loc: Total lines of code
    - lloc: Logical lines of code
    - sloc: Source lines of code (excluding comments and blank lines)
    - comments: Number of comment lines
    - blank: Number of blank lines
    - cyclomatic_complexity: Total cyclomatic complexity score
    - avg_complexity: Average complexity per block/function
    - maintainability_index: Radon Maintainability Index (0-100)
    - halstead_volume: Halstead Volume (code size in bits)
    - halstead_difficulty: Halstead Difficulty (mental effort to inspect)
    - halstead_effort: Halstead Effort (implementation effort)
    - halstead_bugs: Halstead estimated latent bug density
    - syntax_valid: Whether the code could be parsed by Radon

    Args:
        source_code (str): Python source code snippet.

    Returns:
        dict: Computed metrics with safe default fallbacks if parsing fails.
    """
    # Safe defaults in case of syntax failure
    metrics: Dict[str, Any] = {
        "loc": 0,
        "lloc": 0,
        "sloc": 0,
        "comments": 0,
        "blank": 0,
        "cyclomatic_complexity": 1,
        "avg_complexity": 1.0,
        "maintainability_index": 100.0,
        "halstead_volume": 0.0,
        "halstead_difficulty": 0.0,
        "halstead_effort": 0.0,
        "halstead_bugs": 0.0,
        "syntax_valid": True,
    }

    if not source_code or not source_code.strip():
        return metrics

    # 1. Raw Metrics (LOC, LLOC, SLOC)
    try:
        raw_res = analyze(source_code)
        metrics["loc"] = raw_res.loc
        metrics["lloc"] = raw_res.lloc
        metrics["sloc"] = raw_res.sloc
        metrics["comments"] = raw_res.comments
        metrics["blank"] = raw_res.blank
    except Exception:
        lines = source_code.splitlines()
        metrics["loc"] = len(lines)
        metrics["sloc"] = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        metrics["syntax_valid"] = False

    # 2. Cyclomatic Complexity (CC)
    try:
        blocks = cc_visit(source_code)
        if not blocks:
            wrapped = _wrap_in_function(source_code)
            blocks = cc_visit(wrapped)

        if blocks:
            total_cc = sum(b.complexity for b in blocks)
            metrics["cyclomatic_complexity"] = total_cc
            metrics["avg_complexity"] = round(total_cc / len(blocks), 2)
        else:
            metrics["cyclomatic_complexity"] = 1
            metrics["avg_complexity"] = 1.0
    except Exception:
        metrics["cyclomatic_complexity"] = 1
        metrics["avg_complexity"] = 1.0
        metrics["syntax_valid"] = False

    # 3. Maintainability Index (MI)
    try:
        mi = mi_visit(source_code, multi=True)
        metrics["maintainability_index"] = round(mi, 2)
    except Exception:
        try:
            mi = mi_visit(_wrap_in_function(source_code), multi=True)
            metrics["maintainability_index"] = round(mi, 2)
        except Exception:
            metrics["maintainability_index"] = 50.0
            metrics["syntax_valid"] = False

    # 4. Halstead Metrics (Volume, Difficulty, Effort, Bugs)
    try:
        h_res = h_visit(source_code)
        target_h = getattr(h_res, "total", h_res)
        metrics["halstead_volume"] = round(float(getattr(target_h, "volume", 0.0)), 2)
        metrics["halstead_difficulty"] = round(float(getattr(target_h, "difficulty", 0.0)), 2)
        metrics["halstead_effort"] = round(float(getattr(target_h, "effort", 0.0)), 2)
        metrics["halstead_bugs"] = round(float(getattr(target_h, "bugs", 0.0)), 4)
    except Exception:
        try:
            h_res = h_visit(_wrap_in_function(source_code))
            target_h = getattr(h_res, "total", h_res)
            metrics["halstead_volume"] = round(float(getattr(target_h, "volume", 0.0)), 2)
            metrics["halstead_difficulty"] = round(float(getattr(target_h, "difficulty", 0.0)), 2)
            metrics["halstead_effort"] = round(float(getattr(target_h, "effort", 0.0)), 2)
            metrics["halstead_bugs"] = round(float(getattr(target_h, "bugs", 0.0)), 4)
        except Exception:
            pass

    return metrics

