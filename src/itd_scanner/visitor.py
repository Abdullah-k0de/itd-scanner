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
        self.string_vars: Set[str] = set()

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
        2. Converting DataFrame to dict records in loop: df.to_dict('records')
        3. Index-based loop over DataFrame (.iloc[i] / .loc[i])
        4. Index-based loop over Arrays/Tensors/Collections: for i in range(...): arr[i]
        """
        # Form A: Direct method iteration (.iterrows, .itertuples, .iteritems)
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute):
            method_name = node.iter.func.attr
            if method_name in ("iterrows", "itertuples", "iteritems"):
                self._record_finding(
                    "Collection Choker",
                    node.lineno,
                    f"Row-by-row DataFrame iteration using .{method_name}() instead of vectorized operations."
                )
                return

            # Form B: df.to_dict('records') in loop iterator
            if method_name == "to_dict" and node.iter.args:
                arg0 = node.iter.args[0]
                if isinstance(arg0, ast.Constant) and arg0.value == "records":
                    self._record_finding(
                        "Collection Choker",
                        node.lineno,
                        "Converting DataFrame to record dicts ('df.to_dict(\"records\")') in loop iterator; causes high memory overhead and bypasses vectorization."
                    )
                    return

        # Check for index-based looping
        is_range_loop = False
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                is_range_loop = True

        loop_var = node.target.id if isinstance(node.target, ast.Name) else None
        if loop_var:
            for child in ast.walk(node):
                if isinstance(child, ast.Subscript):
                    # Check if subscript uses loop_var as slice index (single or tuple multi-dim)
                    uses_loop_var = False
                    if isinstance(child.slice, ast.Name) and child.slice.id == loop_var:
                        uses_loop_var = True
                    elif isinstance(child.slice, ast.Tuple):
                        for elt in child.slice.elts:
                            if isinstance(elt, ast.Name) and elt.id == loop_var:
                                uses_loop_var = True
                                break

                    if uses_loop_var:
                        # Form C: Explicit .iloc or .loc
                        if isinstance(child.value, ast.Attribute) and child.value.attr in ("iloc", "loc"):
                            self._record_finding(
                                "Collection Choker",
                                child.lineno,
                                f"Inefficient index-based DataFrame iteration using .{child.value.attr}[{loop_var}] inside a loop; use vectorized operations."
                            )
                            return
                        # Form D: Array/Tensor indexing inside a range loop: for i in range(...): arr[i]
                        elif is_range_loop:
                            self._record_finding(
                                "Collection Choker",
                                child.lineno,
                                f"Index-based array/collection iteration ('[{loop_var}]') inside a range loop; bypasses direct iteration and vectorized operations."
                            )
                            return

    # -------------------------------------------------------------------------
    # 2. Math Looper & 3. String Masher (AugAssign / Assign inside loops)
    # -------------------------------------------------------------------------
    def visit_AugAssign(self, node: ast.AugAssign):
        if self.loop_depth > 0:
            # 2. Math Looper: e.g., total += arr[i] or total += x * y or total += x
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow)):
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
                target_is_str = isinstance(node.target, ast.Name) and node.target.id in self.string_vars
                if target_is_str or self._is_probable_string(node.value) or self._is_probable_string(node.target):
                    self._record_finding(
                        "String Masher",
                        node.lineno,
                        "String concatenation using '+=' inside a loop instead of .join() or vectorized str."
                    )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Track string variable initializations at any scope (e.g. s = "" or s = str())
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.string_vars.add(t.id)
        elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "str":
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.string_vars.add(t.id)

        if self.loop_depth > 0:
            # Check DIY Wheel: Manual frequency counter using dict.get(): d[k] = d.get(k, 0) + 1
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                    # Check if one operand is .get(..., 0) and the other is 1
                    is_get_call = False
                    is_increment = False
                    for op_side in (node.value.left, node.value.right):
                        if isinstance(op_side, ast.Call) and isinstance(op_side.func, ast.Attribute) and op_side.func.attr == "get":
                            is_get_call = True
                        elif isinstance(op_side, ast.Constant) and op_side.value == 1:
                            is_increment = True
                    if is_get_call and is_increment:
                        self._record_finding(
                            "DIY Wheel",
                            node.lineno,
                            "Manual frequency counter ('d[k] = d.get(k, 0) + 1'); use collections.Counter() or pd.Series.value_counts()."
                        )

            # Check binary operations in assignment inside loops
            if isinstance(node.value, ast.BinOp):
                math_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        target_id = target.id
                        # Check target = target <op> expr OR target = expr <op> target
                        target_is_left = isinstance(node.value.left, ast.Name) and node.value.left.id == target_id
                        target_is_right = isinstance(node.value.right, ast.Name) and node.value.right.id == target_id

                        if target_is_left or target_is_right:
                            other_side = node.value.right if target_is_left else node.value.left

                            # 3. String Masher (s = s + ... or s = ... + s)
                            if isinstance(node.value.op, ast.Add):
                                if target_id in self.string_vars or self._is_probable_string(other_side):
                                    self._record_finding(
                                        "String Masher",
                                        node.lineno,
                                        "String concatenation using 's = s + ...' inside a loop."
                                    )
                                    continue

                            # 2. Math Looper (total = total <op> expr or total = expr <op> total)
                            if isinstance(node.value.op, math_ops):
                                if self._has_subscript(other_side) or self._has_scalar_math(other_side):
                                    self._record_finding(
                                        "Math Looper",
                                        node.lineno,
                                        f"Accumulator math using '{target_id} = {target_id} <op> ...' inside a loop."
                                    )
                                elif isinstance(other_side, ast.Name) and other_side.id in self.current_loop_vars:
                                    self._record_finding(
                                        "Math Looper",
                                        node.lineno,
                                        f"Manual accumulation of loop variable '{other_side.id}' inside a loop."
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

        # Functions where materializing an intermediate list in memory is wasteful
        reduction_funcs = {"sum", "any", "all", "min", "max", "join", "set", "tuple"}

        if func_name in reduction_funcs and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.ListComp):
                self._record_finding(
                    "RAM Hog",
                    node.lineno,
                    f"Wasteful list comprehension materialized inside '{func_name}()'; use a generator expression or vectorized call."
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
        - Manual frequency counting: `if x in counts: counts[x] += 1` or `d[k] = d.get(k, 0) + 1`
        - Manual conditional counting: `if cond: count += 1` instead of (condition).sum()
        - Manual linear search: `if x == target: found = True; break` instead of `target in data`
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

                        # Pattern E: Manual linear search: if x == target: found = True; break
                        if isinstance(op, ast.Eq):
                            has_break = any(isinstance(s, ast.Break) for s in stmt.body)
                            has_assign = any(isinstance(s, ast.Assign) for s in stmt.body)
                            if has_break and has_assign:
                                self._record_finding(
                                    "DIY Wheel",
                                    stmt.lineno,
                                    "Manual linear search loop ('if x == target: found = True; break'); use 'target in data' or np.isin()."
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
