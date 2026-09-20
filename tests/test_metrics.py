"""
Unit tests for Radon software metrics calculation.
"""

from itd_scanner.metrics import compute_code_metrics

def test_metrics_simple_script():
    code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
total = 0
for idx, row in df.iterrows():
    if row['a'] > 1:
        total += row['a']
"""
    m = compute_code_metrics(code)
    assert m["syntax_valid"] is True
    assert m["loc"] > 0
    assert m["sloc"] > 0
    # The for-loop and if-statement should register in cyclomatic complexity
    assert m["cyclomatic_complexity"] >= 3
    assert m["maintainability_index"] > 0

def test_metrics_empty():
    m = compute_code_metrics("")
    assert m["loc"] == 0
    assert m["cyclomatic_complexity"] == 1

def test_metrics_syntax_error():
    # Should not crash on invalid python syntax
    m = compute_code_metrics("def broken( ::::")
    assert m["syntax_valid"] is False
    assert m["cyclomatic_complexity"] == 1

if __name__ == "__main__":
    test_metrics_simple_script()
    test_metrics_empty()
    test_metrics_syntax_error()
    print("All metrics tests passed successfully!")
