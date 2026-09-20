"""
AST NodeVisitor implementation for detecting Idiomatic Technical Debt (ITD)
anti-patterns in Data Science and Numerical Python code.
"""

import ast
from typing import Dict, List, Any, Optional

class ITDScanner(ast.NodeVisitor):
    """
    Scans an Abstract Syntax Tree (AST) for 5 core Idiomatic Technical Debt smells:
    1. Collection Choker: Inefficient row-by-row iteration over DataFrames/Series.
    2. Math Looper: Pure Python scalar math accumulation loops bypassing vectorized ufuncs.
    3. String Masher: Cumulative string concatenation inside iterative loops.
    4. RAM Hog: Wasteful intermediate list materialization inside reduction calls.
    5. DIY Wheel: Manual re-implementation of built-in/library algorithms (counting, max/min, unique).
    """

    def __init__(self):
        self.flags: Dict[str, bool] = {
            "Collection Choker": False,
            "Math Looper": False,
            "String Masher": False,
            "RAM Hog": False,
            "DIY Wheel": False,
        }
        self.findings: List[Dict[str, Any]] = []
        self.loop_depth = 0
        self.current_loop_target: Optional[str] = None

    def _record_finding(self, smell: str, lineno: int, message: str):
        self.flags[smell] = True
        self.findings.append({
            "smell": smell,
            "line": lineno,
            "message": message,
        })

    # -------------------------------------------------------------------------
    # Loop tracking
    # -------------------------------------------------------------------------
    def visit_For(self, node: ast.For):
        # 1. Check for Collection Choker in loop iterator
        self._check_collection_choker(node)

        # 2. Check for DIY Wheel patterns inside the loop
        self._check_diy_wheel(node)

        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    # -------------------------------------------------------------------------
    # 1. Collection Choker
    # -------------------------------------------------------------------------
    def _check_collection_choker(self, node: ast.For):
        """
        Flags calls to .iterrows(), .itertuples(), .iteritems(), .items()
        on DataFrames/Series in a for-loop header.
        """
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute):
            method_name = node.iter.func.attr
            if method_name in ("iterrows", "itertuples", "iteritems"):
                self._record_finding(
                    "Collection Choker",
                    node.lineno,
                    f"Row-by-row DataFrame iteration using .{method_name}() instead of vectorized operations."
                )

    # -------------------------------------------------------------------------
    # 2. Math Looper & 3. String Masher (AugAssign / Assign inside loops)
    # -------------------------------------------------------------------------
    def visit_AugAssign(self, node: ast.AugAssign):
        if self.loop_depth > 0:
            # 2. Math Looper: e.g., total += arr[i] or total += x * y
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                # Check if RHS contains subscript (indexing like arr[i], row['val']) or arithmetic
                if self._has_subscript(node.value) or self._has_scalar_math(node.value):
                    self._record_finding(
                        "Math Looper",
                        node.lineno,
                        "Accumulator math inside loop over indexed elements bypassing vectorized ufuncs."
                    )

            # 3. String Masher: string concatenation (s += ...)
            if isinstance(node.op, ast.Add):
                if self._is_probable_string(node.value) or self._is_probable_string(node.target):
                    self._record_finding(
                        "String Masher",
                        node.lineno,
                        "String concatenation using '+=' inside a loop instead of .join() or vectorized str."
                    )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        if self.loop_depth > 0:
            # Check s = s + ... (String Masher alternative syntax)
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # s = s + ...
                        if isinstance(node.value.left, ast.Name) and node.value.left.id == target.id:
                            if self._is_probable_string(node.value.right):
                                self._record_finding(
                                    "String Masher",
                                    node.lineno,
                                    "String concatenation using 's = s + ...' inside a loop."
                                )
                        # total = total + arr[i] (Math Looper alternative syntax)
                        elif isinstance(node.value.left, ast.Name) and node.value.left.id == target.id:
                            if self._has_subscript(node.value.right) or self._has_scalar_math(node.value.right):
                                self._record_finding(
                                    "Math Looper",
                                    node.lineno,
                                    "Accumulator math using 'x = x + ...' inside a loop."
                                )

        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # 4. RAM Hog (Wasteful list comprehensions in reduction calls)
    # -------------------------------------------------------------------------
    def visit_Call(self, node: ast.Call):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Functions that consume iterables and do not require a materialized list
        reduction_funcs = {"sum", "any", "all", "min", "max", "join"}

        if func_name in reduction_funcs and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.ListComp):
                self._record_finding(
                    "RAM Hog",
                    node.lineno,
                    f"Wasteful list comprehension materialized inside '{func_name}()'; use a generator expression."
                )

        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # 5. DIY Wheel (Reinventing built-in algorithms)
    # -------------------------------------------------------------------------
    def _check_diy_wheel(self, loop_node: ast.For):
        """
        Detects common manual reinventions inside loops:
        - Manual frequency counting: `if x in counts: counts[x] += 1` instead of collections.Counter
        - Manual unique tracking: `if x not in seen: seen.append(x)` instead of set() or pd.unique()
        - Manual max/min search: `if x > max_val: max_val = x` instead of max()
        """
        for stmt in loop_node.body:
            if isinstance(stmt, ast.If):
                # Pattern A: Manual unique: if x not in seen: seen.append(x)
                if isinstance(stmt.test, ast.Compare):
                    for op in stmt.test.ops:
                        if isinstance(op, ast.NotIn):
                            # Look for append inside body
                            for sub_stmt in stmt.body:
                                if isinstance(sub_stmt, ast.Expr) and isinstance(sub_stmt.value, ast.Call):
                                    if isinstance(sub_stmt.value.func, ast.Attribute) and sub_stmt.value.func.attr in ("append", "add"):
                                        self._record_finding(
                                            "DIY Wheel",
                                            stmt.lineno,
                                            "Manual unique collection reinvention ('if x not in seen: seen.append(x)'); use set() or pd.unique()."
                                        )
                                        return

                        # Pattern B: Manual max/min search: if x > current_max: current_max = x
                        if isinstance(op, (ast.Gt, ast.Lt, ast.GtE, ast.LtE)):
                            for sub_stmt in stmt.body:
                                if isinstance(sub_stmt, ast.Assign):
                                    self._record_finding(
                                        "DIY Wheel",
                                        stmt.lineno,
                                        "Manual min/max comparison loop; use built-in min(), max(), or np.min()/np.max()."
                                    )
                                    return

                        # Pattern C: Manual frequency counter: if x in d: d[x] += 1 else: d[x] = 1
                        if isinstance(op, ast.In):
                            for sub_stmt in stmt.body:
                                if isinstance(sub_stmt, ast.AugAssign) and isinstance(sub_stmt.target, ast.Subscript):
                                    self._record_finding(
                                        "DIY Wheel",
                                        stmt.lineno,
                                        "Manual frequency counter loop; use collections.Counter() or pd.Series.value_counts()."
                                    )
                                    return

    # -------------------------------------------------------------------------
    # Helper methods for AST inspection
    # -------------------------------------------------------------------------
    def _has_subscript(self, node: ast.AST) -> bool:
        """Returns True if the expression accesses an index or slice (e.g. arr[i], df['c'])."""
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                return True
        return False

    def _has_scalar_math(self, node: ast.AST) -> bool:
        """Returns True if the expression contains binary math operations."""
        for child in ast.walk(node):
            if isinstance(child, (ast.BinOp, ast.UnaryOp)):
                return True
        return False

    def _is_probable_string(self, node: ast.AST) -> bool:
        """
        Heuristic check for string concatenation operands:
        string literals, str() calls, .format(), or f-strings.
        Crucially ignores string constants used as dictionary/DataFrame column keys (e.g., df['col']).
        """
        def check(n: ast.AST) -> bool:
            if isinstance(n, ast.Subscript):
                # Inspect the base container (e.g. s[i]), but skip the slice key (e.g. 'col_name')
                return check(n.value)
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                return True
            if isinstance(n, ast.JoinedStr):
                return True
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name) and n.func.id == "str":
                    return True
                if isinstance(n.func, ast.Attribute) and n.func.attr in ("format", "lower", "upper", "strip", "join"):
                    return True
            # Recurse through children
            for child in ast.iter_child_nodes(n):
                if check(child):
                    return True
            return False

        return check(node)

