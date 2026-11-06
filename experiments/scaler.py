"""
Phase 3 - Step 2 (Phase B): Universal Synthetic Data Scaling Engine (`scaler.py`).

Responsibilities:
1. Normalize standalone LLM/Reference scripts (flattening `if __name__ == '__main__':`
   and inlining `main()` when used as a wrapper).
2. Partition each script AST into:
   - `setup_code`: Imports, helper function/class definitions, random seeds, and initial
     input data creation (including everything preparing inputs prior to a solver function call).
   - `exec_code`: The actual algorithm / loop / vectorized transformation under test.
   - Auto-Invocation of Uncalled Functions: If a script defines a solver function (e.g.,
     `def calculate_slopes(df):`) without a top-level call statement, automatically loads
     `_raw_test_input = load_data()` in `setup_code`, scales it in `scale_environment()`,
     and invokes the function on `_raw_test_input` in `exec_code`.
3. Provide `scale_environment(env, target_n)` to dynamically scale initialized input
   structures across the 3 Phase 3 blueprint tiers:
   - Tier 1 (Base/Baseline): 10^3 elements (1,000)
   - Tier 2 (Stress Test):   10^5 elements (100,000)
   - Tier 3 (Stress Ceiling): 10^6 elements (1,000,000)
   while strictly preserving column schemas, data types, index structures, and paired
   permutation/index arrays across Pandas, NumPy, PyTorch, and pure-Python containers.
"""

from __future__ import annotations

import ast
import copy
import io
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Phase 3 Blueprint Tiers
TIER_SIZES: Dict[int, int] = {
    1: 1_000,      # Tier 1 (Base/Baseline): 10^3 elements
    2: 100_000,    # Tier 2 (Stress Test): 10^5 elements
    3: 1_000_000,  # Tier 3 (Massive Array / Stress Ceiling): 10^6 elements
}

# Constructors recognized as initial input creation calls
DATA_CREATION_CALLS: Set[str] = {
    "DataFrame",
    "Series",
    "read_csv",
    "read_table",
    "read_fwf",
    "date_range",
    "Timestamp",
    "array",
    "asarray",
    "arange",
    "linspace",
    "zeros",
    "ones",
    "empty",
    "full",
    "eye",
    "rand",
    "randn",
    "randint",
    "random",
    "choice",
    "permutation",
    "uniform",
    "normal",
    "exponential",
    "poisson",
    "binomial",
    "beta",
    "gamma",
    "seed",
    "manual_seed",
    "default_rng",
    "RandomState",
    "tensor",
    "FloatTensor",
    "LongTensor",
    "TensorDataset",
    "DataLoader",
    "load_data",
    "StringIO",
    "make_classification",
    "make_regression",
    "make_blobs",
}


def _extract_call_names(node: ast.AST) -> Set[str]:
    """Return all function/method names called inside an AST subtree."""
    names: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                names.add(sub.func.id)
            elif isinstance(sub.func, ast.Attribute):
                names.add(sub.func.attr)
    return names


def _is_pure_literal_or_constructor(
    stmt: ast.stmt,
    user_funcs: Set[str],
    initialized_data_vars: Set[str],
) -> bool:
    """
    Determine whether a top-level statement is part of initial input data setup
    rather than the algorithm execution phase.
    """
    if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return True

    # Allow standalone seed calls (e.g. `np.random.seed(1)`, `random.seed(42)`, `torch.manual_seed(0)`)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        expr_calls = _extract_call_names(stmt.value)
        if expr_calls and expr_calls.issubset({"seed", "manual_seed", "set_seed"}):
            return True
        return False

    # Never treat top-level For/While/Try/With blocks as static data initialization
    if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
        return False

    if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        return False

    value = stmt.value
    if value is None:
        return True

    # If the assignment calls a user-defined function (e.g., `result = g(df)`),
    # this is where algorithm execution begins!
    called_names = _extract_call_names(value)
    if called_names & user_funcs:
        return False

    # Check if there is any loop/comprehension inside the assignment
    has_comp = any(
        isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
        for n in ast.walk(value)
    )
    if has_comp:
        # Allow tiny coordinate-grid setups in reference solutions like
        # `points1 = np.array([(x, y) for x in np.linspace(-1, 1, 7) ...])`
        # ONLY if no data variable has been initialized yet.
        if initialized_data_vars or not (called_names & {"array", "linspace", "DataFrame", "arange"}):
            return False

    # Check if the RHS references an already-initialized data variable to transform it
    referenced_names = {
        n.id for n in ast.walk(value) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }

    # If RHS is purely literals (numbers, strings, lists, dicts, tuples, sets) with no variable loads:
    if not referenced_names:
        return True

    # If RHS calls a known data constructor (e.g. `df = pd.DataFrame(data)`, `x1 = np.random.normal(..., size=n_samples)`):
    if called_names and called_names.issubset(
        DATA_CREATION_CALLS | {"len", "list", "dict", "set", "tuple", "range", "int", "float", "str"}
    ):
        for sub in ast.walk(value):
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id in initialized_data_vars:
                return False
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id in initialized_data_vars:
                if sub.attr not in {"shape", "columns", "index", "dtype", "ndim"}:
                    return False
        return True

    return False


def normalize_and_split_script(code: str) -> Tuple[str, str]:
    """
    Normalize a standalone Python script (unwrapping `if __name__ == '__main__':`
    and inlining `def main():` if present) and partition it into `(setup_code, exec_code)`.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        cleaned = code.expandtabs(4)
        tree = ast.parse(cleaned)

    # Step 1: Collect top-level function definitions and check if `main()` is used
    main_func_body: Optional[List[ast.stmt]] = None
    flattened_body: List[ast.stmt] = []

    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "main" and len(stmt.args.args) == 0:
            main_func_body = stmt.body
            continue
        if (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and any(isinstance(n, ast.Name) and n.id == "__name__" for n in ast.walk(stmt.test))
        ):
            for inner_stmt in stmt.body:
                if (
                    isinstance(inner_stmt, ast.Expr)
                    and isinstance(inner_stmt.value, ast.Call)
                    and isinstance(inner_stmt.value.func, ast.Name)
                    and inner_stmt.value.func.id == "main"
                    and main_func_body is not None
                ):
                    flattened_body.extend(main_func_body)
                else:
                    flattened_body.append(inner_stmt)
            continue
        flattened_body.append(stmt)

    final_body: List[ast.stmt] = []
    for stmt in flattened_body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "main"
            and main_func_body is not None
        ):
            final_body.extend(main_func_body)
        else:
            final_body.append(stmt)

    class _TopLevelReturnCleaner(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            return node

        def visit_Return(self, node: ast.Return) -> ast.AST:
            if node.value is not None:
                return ast.Assign(
                    targets=[ast.Name(id="_main_return_val", ctx=ast.Store())],
                    value=node.value,
                )
            return ast.Pass()

    cleaner = _TopLevelReturnCleaner()
    final_body = [ast.fix_missing_locations(cleaner.visit(s)) for s in final_body]

    # Step 2: Check if there is a top-level solver function that is called at module level.
    # If a user-defined function containing loops/algorithm logic is called at statement `first_call_idx`,
    # everything prior to `first_call_idx` (excluding loops over already-created data) is input setup!
    user_func_nodes: List[ast.FunctionDef] = [
        s for s in final_body if isinstance(s, ast.FunctionDef)
    ]
    user_funcs: Set[str] = {
        s.name for s in final_body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    first_user_func_call_idx: Optional[int] = None
    for idx, stmt in enumerate(final_body):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        called = _extract_call_names(stmt)
        if called & {f.name for f in user_func_nodes}:
            first_user_func_call_idx = idx
            break

    setup_stmts: List[ast.stmt] = []
    exec_stmts: List[ast.stmt] = []
    initialized_data_vars: Set[str] = set()
    in_exec_phase = False

    if first_user_func_call_idx is not None:
        # Check if there are any top-level For/While loops before first_user_func_call_idx
        has_top_loop_before_call = any(
            isinstance(final_body[i], (ast.For, ast.While)) for i in range(first_user_func_call_idx)
        )
        if not has_top_loop_before_call:
            for idx, stmt in enumerate(final_body):
                if idx < first_user_func_call_idx or isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    # Skip pure print statements in setup
                    if isinstance(stmt, ast.Expr) and _extract_call_names(stmt) == {"print"}:
                        continue
                    setup_stmts.append(stmt)
                else:
                    exec_stmts.append(stmt)
            setup_mod = ast.Module(body=setup_stmts, type_ignores=[])
            exec_mod = ast.Module(body=exec_stmts, type_ignores=[])
            return ast.unparse(setup_mod), ast.unparse(exec_mod)

    # Standard statement-by-statement partitioning
    for stmt in final_body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            setup_stmts.append(stmt)
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not in_exec_phase:
            setup_stmts.append(stmt)
            continue

        if not in_exec_phase and _is_pure_literal_or_constructor(stmt, user_funcs, initialized_data_vars):
            setup_stmts.append(stmt)
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            initialized_data_vars.add(n.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                initialized_data_vars.add(stmt.target.id)
        else:
            in_exec_phase = True
            exec_stmts.append(stmt)

    # Step 3: Handle scripts where `exec_stmts` is empty because the LLM ONLY defined a function
    # (e.g. `def calculate_slopes(df): ...`) and never called it!
    if not exec_stmts and user_func_nodes:
        target_func = user_func_nodes[-1]
        fn_name = target_func.name
        setup_code_str = ast.unparse(ast.Module(body=setup_stmts, type_ignores=[]))
        # Load `_raw_test_input` from DS-1000's `load_data()` hook during setup so `scale_environment` scales it!
        setup_code_str += "\n_raw_test_input = load_data() if 'load_data' in dir() else None\n"
        exec_code_str = (
            f"if '_raw_test_input' in dir() and _raw_test_input is not None:\n"
            f"    if isinstance(_raw_test_input, tuple):\n"
            f"        try:\n"
            f"            _auto_res = {fn_name}(*_raw_test_input)\n"
            f"        except TypeError:\n"
            f"            _auto_res = {fn_name}(_raw_test_input[0])\n"
            f"    else:\n"
            f"        _auto_res = {fn_name}(_raw_test_input)\n"
            f"else:\n"
            f"    _auto_res = {fn_name}()\n"
        )
        return setup_code_str, exec_code_str

    # Edge case: if all statements ended up in setup_stmts (e.g., a 2-line script `a = np.array(...); result = a.sum()`),
    # move the last non-import statement into exec_stmts so exec_code is never empty.
    if not exec_stmts and setup_stmts:
        for idx in range(len(setup_stmts) - 1, -1, -1):
            if not isinstance(setup_stmts[idx], (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                exec_stmts.insert(0, setup_stmts.pop(idx))
                break

    setup_mod = ast.Module(body=setup_stmts, type_ignores=[])
    exec_mod = ast.Module(body=exec_stmts, type_ignores=[])
    return ast.unparse(setup_mod), ast.unparse(exec_mod)


# ==============================================================================
# Universal Object & Environment Scaling Functions
# ==============================================================================

def scale_dataframe(df: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """Scale a Pandas DataFrame along axis 0 to `target_n` rows while preserving schema & types."""
    n_rows = len(df)
    if n_rows == 0 or n_rows >= target_n:
        return df

    repeats = math.ceil(target_n / n_rows)
    orig_index = df.index
    scaled = pd.concat([df] * repeats, ignore_index=True).iloc[:target_n].copy()

    if isinstance(orig_index, pd.DatetimeIndex):
        start = orig_index[0] if len(orig_index) > 0 else pd.Timestamp("2020-01-01")
        freq = orig_index.freq or "s"
        scaled.index = pd.date_range(start=start, periods=target_n, freq=freq, tz=orig_index.tz)
        scaled.index.name = orig_index.name
    elif isinstance(orig_index, pd.MultiIndex):
        tiled_tuples = (list(orig_index) * repeats)[:target_n]
        scaled.index = pd.MultiIndex.from_tuples(tiled_tuples, names=orig_index.names)
    else:
        scaled.index.name = orig_index.name

    return scaled


def scale_series(s: pd.Series, target_n: int) -> pd.Series:
    """Scale a Pandas Series along axis 0 to `target_n` elements."""
    n_rows = len(s)
    if n_rows == 0 or n_rows >= target_n:
        return s

    repeats = math.ceil(target_n / n_rows)
    orig_index = s.index
    scaled = pd.concat([s] * repeats, ignore_index=True).iloc[:target_n].copy()
    if isinstance(orig_index, pd.DatetimeIndex):
        start = orig_index[0] if len(orig_index) > 0 else pd.Timestamp("2020-01-01")
        scaled.index = pd.date_range(start=start, periods=target_n, freq="s", tz=orig_index.tz)
    scaled.name = s.name
    scaled.index.name = orig_index.name
    return scaled


def scale_ndarray(arr: np.ndarray, target_n: int) -> np.ndarray:
    """Scale a NumPy ndarray along axis 0 to `target_n` elements."""
    if arr.ndim == 0 or arr.shape[0] == 0 or arr.shape[0] >= target_n:
        return arr

    n0 = arr.shape[0]
    repeats = math.ceil(target_n / n0)
    tile_spec = (repeats,) + (1,) * (arr.ndim - 1)
    scaled = np.tile(arr, tile_spec)[:target_n].copy()
    return scaled


def scale_tensor(t: "torch.Tensor", target_n: int) -> "torch.Tensor":
    """Scale a PyTorch Tensor along dim 0 to `target_n` elements."""
    if not HAS_TORCH or t.ndim == 0 or t.shape[0] == 0 or t.shape[0] >= target_n:
        return t

    n0 = t.shape[0]
    repeats = math.ceil(target_n / n0)
    repeat_spec = (repeats,) + (1,) * (t.ndim - 1)
    return t.repeat(*repeat_spec)[:target_n].clone()


def _is_permutation_of_n(seq: Any, orig_n: int) -> bool:
    """Check if `seq` (list or 1D ndarray) is an integer index permutation of `0..orig_n-1`."""
    if orig_n < 2:
        return False
    if isinstance(seq, list) and len(seq) == orig_n and all(isinstance(x, int) for x in seq):
        return set(seq) == set(range(orig_n))
    if isinstance(seq, np.ndarray) and seq.ndim == 1 and len(seq) == orig_n and np.issubdtype(seq.dtype, np.integer):
        return set(seq.tolist()) == set(range(orig_n))
    return False


def inject_load_data_hook(env: Dict[str, Any], code_context: Optional[str] = None) -> None:
    """
    Inject a `load_data()` helper into `env` using DS-1000's `code_context` (`generate_test_case(1)`).
    """
    if not code_context or not isinstance(code_context, str):
        return

    def _load_data() -> Any:
        ctx_env: Dict[str, Any] = {}
        exec(code_context, ctx_env)
        if "generate_test_case" in ctx_env:
            test_input, _ = ctx_env["generate_test_case"](1)
            return test_input
        if "define_test_input" in ctx_env:
            return ctx_env["define_test_input"](1)
        return None

    env["load_data"] = _load_data


def scale_single_object(obj: Any, target_n: int) -> Any:
    """Scale an individual DS-1000 `test_input` object (including tuples of arrays/tensors)."""
    if isinstance(obj, pd.DataFrame):
        return scale_dataframe(obj, target_n)
    if isinstance(obj, pd.Series):
        return scale_series(obj, target_n)
    if isinstance(obj, np.ndarray) and obj.ndim >= 1 and obj.shape[0] >= 2:
        return scale_ndarray(obj, target_n)
    if HAS_TORCH and isinstance(obj, torch.Tensor) and obj.ndim >= 1 and obj.shape[0] >= 2:
        return scale_tensor(obj, target_n)
    if isinstance(obj, tuple):
        return tuple(scale_single_object(item, target_n) for item in obj)
    if isinstance(obj, list) and len(obj) >= 2:
        repeats = math.ceil(target_n / len(obj))
        return (obj * repeats)[:target_n]
    return obj


def scale_environment(env: Dict[str, Any], target_n: int) -> Dict[str, Any]:
    """
    Inspect a script's post-setup namespace `env` and scale all input data structures
    along their primary axis to `target_n` while keeping paired variables synchronized.
    """
    user_keys = [
        k for k in env.keys()
        if not k.startswith("__")
        and k not in {"pd", "np", "torch", "scipy", "sklearn", "io", "re", "math", "copy", "itertools", "collections", "csv", "datetime"}
        and not callable(env[k])
        and not isinstance(env[k], type)
    ]

    orig_lengths: Set[int] = set()
    has_rich_containers = False

    # Pass 1: Scale Pandas DataFrames, Series, NumPy arrays, PyTorch Tensors, and Tuples from load_data()
    for k in user_keys:
        val = env[k]
        if isinstance(val, pd.DataFrame) and len(val) >= 2:
            orig_lengths.add(len(val))
            env[k] = scale_dataframe(val, target_n)
            has_rich_containers = True
        elif isinstance(val, pd.Series) and len(val) >= 2:
            orig_lengths.add(len(val))
            env[k] = scale_series(val, target_n)
            has_rich_containers = True
        elif isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] >= 2:
            orig_lengths.add(val.shape[0])
            env[k] = scale_ndarray(val, target_n)
            has_rich_containers = True
        elif HAS_TORCH and isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] >= 2:
            orig_lengths.add(val.shape[0])
            env[k] = scale_tensor(val, target_n)
            has_rich_containers = True
        elif isinstance(val, tuple) and any(
            isinstance(x, (pd.DataFrame, pd.Series, np.ndarray)) or (HAS_TORCH and isinstance(x, torch.Tensor))
            for x in val
        ):
            env[k] = scale_single_object(val, target_n)
            has_rich_containers = True

    # Pass 1b: Rebuild PyTorch TensorDataset / DataLoader if present
    if HAS_TORCH:
        from torch.utils.data import DataLoader, TensorDataset
        for k in user_keys:
            val = env[k]
            if isinstance(val, TensorDataset) and len(val.tensors) > 0:
                scaled_tensors = [scale_tensor(t, target_n) for t in val.tensors]
                env[k] = TensorDataset(*scaled_tensors)
                has_rich_containers = True
        for k in user_keys:
            val = env[k]
            if isinstance(val, DataLoader) and isinstance(val.dataset, TensorDataset):
                scaled_tensors = [scale_tensor(t, target_n) for t in val.dataset.tensors]
                new_ds = TensorDataset(*scaled_tensors)
                env[k] = DataLoader(new_ds, batch_size=val.batch_size or 32, shuffle=False)

    # Pass 2: Synchronize any row-permutation index lists and explicit `n_samples` / `N` counters
    for k in user_keys:
        val = env[k]
        for orig_n in orig_lengths:
            if _is_permutation_of_n(val, orig_n):
                rng = np.random.default_rng(42)
                perm = rng.permutation(target_n)
                env[k] = perm.tolist() if isinstance(val, list) else perm
            elif isinstance(val, int) and val == orig_n and k in {"N", "n", "n_samples", "num_rows", "rows", "length"}:
                env[k] = target_n

    # Pass 3: Pure-Python `Negative` scripts (no DataFrame/ndarray/Tensor created)
    if not has_rich_containers:
        for k in user_keys:
            val = env[k]
            if isinstance(val, dict) and len(val) >= 1:
                list_vals = [v for v in val.values() if isinstance(v, list) and len(v) >= 2]
                if len(list_vals) == len(val):
                    new_dict = {}
                    for dk, dv in val.items():
                        repeats = math.ceil(target_n / len(dv))
                        new_dict[dk] = (dv * repeats)[:target_n]
                    env[k] = new_dict
                    orig_lengths.add(len(list_vals[0]))
                    continue

            if isinstance(val, list) and len(val) >= 2:
                if all(isinstance(x, list) and len(x) >= 4 for x in val) and len(val) <= 5 and len(val[0]) > len(val):
                    col_len = len(val[0])
                    if all(len(x) == col_len for x in val):
                        repeats = math.ceil(target_n / col_len)
                        env[k] = [(col * repeats)[:target_n] for col in val]
                        continue
                skip_names = {
                    "header", "headers", "columns", "cols", "column_names", "keys",
                    "month_names", "months", "days", "day_names", "alphabet", "digits", "vowels",
                }
                if k.lower() not in skip_names:
                    repeats = math.ceil(target_n / len(val))
                    env[k] = (val * repeats)[:target_n]

    return env
