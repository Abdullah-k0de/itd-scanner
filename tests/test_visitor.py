"""
Unit tests for ITDScanner AST visitor.
"""

from itd_scanner import scan_code

def test_collection_choker():
    code_bad = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
total = 0
for idx, row in df.iterrows():
    total += row['a']
"""
    result = scan_code(code_bad)
    assert result["Collection Choker"] is True

    code_good = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
total = df['a'].sum()
"""
    result_good = scan_code(code_good)
    assert result_good["Collection Choker"] is False

def test_math_looper():
    code_bad = """
total = 0
for i in range(len(arr)):
    total += arr[i] * 2.5
"""
    result = scan_code(code_bad)
    assert result["Math Looper"] is True

    code_good = """
import numpy as np
total = np.sum(arr * 2.5)
"""
    result_good = scan_code(code_good)
    assert result_good["Math Looper"] is False

def test_string_masher():
    code_bad = """
s = ""
for w in words:
    s += str(w) + ", "
"""
    result = scan_code(code_bad)
    assert result["String Masher"] is True

    code_good = """
s = ", ".join(words)
"""
    result_good = scan_code(code_good)
    assert result_good["String Masher"] is False

def test_ram_hog():
    code_bad = """
total = sum([x * 2 for x in range(1000)])
"""
    result = scan_code(code_bad)
    assert result["RAM Hog"] is True

    code_good = """
total = sum(x * 2 for x in range(1000))
"""
    result_good = scan_code(code_good)
    assert result_good["RAM Hog"] is False

def test_diy_wheel():
    code_bad = """
seen = []
for item in dataset:
    if item not in seen:
        seen.append(item)
"""
    result = scan_code(code_bad)
    assert result["DIY Wheel"] is True

    code_good = """
seen = list(set(dataset))
"""
    result_good = scan_code(code_good)
    assert result_good["DIY Wheel"] is False

def test_collection_choker_array_index():
    code = """
import numpy as np
res = []
for i in range(len(arr)):
    res.append(arr[i])
"""
    result = scan_code(code)
    assert result["Collection Choker"] is True

def test_collection_choker_2d_matrix():
    code = """
res = []
for i in range(n):
    res.append(matrix[i, 0])
"""
    result = scan_code(code)
    assert result["Collection Choker"] is True

def test_collection_choker_to_dict_records():
    code = """
for row in df.to_dict('records'):
    process(row)
"""
    result = scan_code(code)
    assert result["Collection Choker"] is True

def test_math_looper_assign_product():
    code = """
prod = 1
for x in factors:
    prod = prod * x
"""
    result = scan_code(code)
    assert result["Math Looper"] is True

def test_math_looper_commutative():
    code = """
total = 0
for x in numbers:
    total = x + total
"""
    result = scan_code(code)
    assert result["Math Looper"] is True

def test_string_masher_tracked_var():
    code = """
res = ""
for word in words:
    res += word
"""
    result = scan_code(code)
    assert result["String Masher"] is True

def test_ram_hog_set():
    code = """
unique_items = set([x.strip() for x in raw_strings])
"""
    result = scan_code(code)
    assert result["RAM Hog"] is True

def test_diy_wheel_dict_get():
    code = """
counts = {}
for item in items:
    counts[item] = counts.get(item, 0) + 1
"""
    result = scan_code(code)
    assert result["DIY Wheel"] is True

def test_diy_wheel_linear_search():
    code = """
found = False
for item in dataset:
    if item == target:
        found = True
        break
"""
    result = scan_code(code)
    assert result["DIY Wheel"] is True

def test_syntax_error_safety():
    code_invalid = "def broken_python_syntax( :"
    # Must not crash, should return all False flags
    result = scan_code(code_invalid)
    assert all(flag is False for flag in result.values())

if __name__ == "__main__":
    test_collection_choker()
    test_collection_choker_array_index()
    test_collection_choker_2d_matrix()
    test_collection_choker_to_dict_records()
    test_math_looper()
    test_math_looper_assign_product()
    test_math_looper_commutative()
    test_string_masher()
    test_string_masher_tracked_var()
    test_ram_hog()
    test_ram_hog_set()
    test_diy_wheel()
    test_diy_wheel_dict_get()
    test_diy_wheel_linear_search()
    test_syntax_error_safety()
    print("All tests passed successfully!")

