"""
AST NodeVisitor implementation for detecting Idiomatic Technical Debt (ITD)
anti-patterns in Data Science and Numerical Python code.
"""

import ast
from typing import Dict, List, Any, Optional, Set

class ITDScanner(ast.NodeVisitor):
    """
    Scans an Abstract Syntax Tree (AST) for 5 core Idiomatic Technical Debt smells:
    1. Collection Choker: Inefficient row-by-row iteration over DataFrames/Series (.iterrows, .itertuples, .iloc[i]).
    2. Math Looper: Pure Python scalar math accumulation loops bypassing vectorized ufuncs.
    3. String Masher: Cumulative string concatenation inside iterative loops.
    4. RAM Hog: Wasteful intermediate list materialization inside reduction calls.
    5. DIY Wheel: Manual re-implementation of built-in/library algorithms (counting, conditional sum, max/min, unique).
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
        self.current_loop_vars: Set[str] = set()

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
        # Track loop target variable names
        added_vars = set()
        if isinstance(node.target, ast.Name):
            added_vars.add(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    added_vars.add(elt.id)

        self.current_loop_vars.update(added_vars)

        # 1. Check for Collection Choker
        self._check_collection_choker(node)

        # 2. Check for DIY Wheel patterns
        self._check_diy_wheel(node)

        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

        self.current_loop_vars.difference_update(added_vars)

    def visit_While(self, node: ast.While):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    # -------------------------------------------------------------------------
    # 1. Collection Choker
    # -------------------------------------------------------------------------
    def _check_collection_choker(self, node: ast.For):
        """
        Flags:
        1. Direct iterator calls: .iterrows(), .itertuples(), .iteritems()
        2. Index-based loop over DataFrame: for i in range(...): ... df.iloc[i] or df.loc[i]
        """
        # Form A: Direct method iteration
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute):
            method_name = node.iter.func.attr
            if method_name in ("iterrows", "itertuples", "iteritems"):
                self._record_finding(
                    "Collection Choker",
                    node.lineno,
                    f"Row-by-row DataFrame iteration using .{method_name}() instead of vectorized operations."
                )
                return

        # Form B: Index-based loop over DataFrame (.iloc[i] / .loc[i])
        loop_var = node.target.id if isinstance(node.target, ast.Name) else None
        if loop_var:
            for child in ast.walk(node):
                if isinstance(child, ast.Subscript):
                    if isinstance(child.slice, ast.Name) and child.slice.id == loop_var:
                        if isinstance(child.value, ast.Attribute) and child.value.attr in ("iloc", "loc"):
                            self._record_finding(
                                "Collection Choker",
                                child.lineno,
                                f"Inefficient index-based DataFrame iteration using .{child.value.attr}[{loop_var}] inside a loop; use vectorized operations."
                            )
                            return

    # -------------------------------------------------------------------------
    # 2. Math Looper & 3. String Masher (AugAssign / Assign inside loops)
    # -------------------------------------------------------------------------
    def visit_AugAssign(self, node: ast.AugAssign):
        if self.loop_depth > 0:
            # 2. Math Looper: e.g., total += arr[i] or total += x * y or total += x
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                # If RHS is a scalar math expression or indexes into a collection
                if self._has_subscript(node.value) or self._has_scalar_math(node.value):
                    self._record_finding(
                        "Math Looper",
                        node.lineno,
                        "Accumulator math inside loop over indexed elements bypassing vectorized ufuncs."
                    )
                # If accumulating loop variable directly (e.g. total += x)
                elif isinstance(node.value, ast.Name) and node.value.id in self.current_loop_vars:
                    self._record_finding(
                        "Math Looper",
                        node.lineno,
                        f"Manual accumulation of loop variable '{node.value.id}' instead of .sum() or np.sum()."
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
        - Manual unique tracking: `if x not in seen: seen.append(x)` instead of set() or pd.unique()
        - Manual min/max search: `if x > max_val: max_val = x` instead of max()
        - Manual frequency counting: `if x in counts: counts[x] += 1` instead of Counter or value_counts()
        - Manual conditional counting: `if cond: count += 1` instead of (condition).sum()
        """
        for stmt in loop_node.body:
            if isinstance(stmt, ast.If):
                # Pattern A: Manual unique: if x not in seen: seen.append(x)
                if isinstance(stmt.test, ast.Compare):
                    for op in stmt.test.ops:
                        if isinstance(op, ast.NotIn):
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

                        # Pattern C: Manual frequency counter: if x in d: d[x] += 1
                        if isinstance(op, ast.In):
                            for sub_stmt in stmt.body:
                                if isinstance(sub_stmt, ast.AugAssign) and isinstance(sub_stmt.target, ast.Subscript):
                                    self._record_finding(
                                        "DIY Wheel",
                                        stmt.lineno,
                                        "Manual frequency counter loop; use collections.Counter() or pd.Series.value_counts()."
                                    )
                                    return

                # Pattern D: Manual conditional counter: if condition: count += 1 (instead of (condition).sum())
                for sub_stmt in stmt.body:
                    if isinstance(sub_stmt, ast.AugAssign) and isinstance(sub_stmt.op, ast.Add):
                        if isinstance(sub_stmt.value, ast.Constant) and sub_stmt.value.value == 1:
                            self._record_finding(
                                "DIY Wheel",
                                stmt.lineno,
                                "Manual conditional counter loop ('if cond: count += 1'); use vectorized mask summation (condition).sum()."
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
            for child in ast.iter_child_nodes(n):
                if check(child):
                    return True
            return False

        return check(node)
