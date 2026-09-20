"""
Baseline linter wrappers for comparative evaluation:
- dslinter (SERG-Delft Pylint plugin for ML/DS code smells)
- perflint (Pylint plugin for performance anti-patterns)
- pandas-vet (Flake8 plugin for Pandas best practices)
"""

import os
import sys
import re
import tempfile
import subprocess
from typing import Dict, List, Any

# Synthetic import header so external linters don't complain about undefined names (e.g. df, np, pd)
SYNTHETIC_HEADER = """
import pandas as pd
import numpy as np
"""

def _run_command_on_code(code: str, cmd_template: List[str], inject_header: bool = True, timeout_sec: int = 15) -> Dict[str, Any]:
    """
    Writes code to a temporary python file, executes the command, and captures output.
    """
    full_code = (SYNTHETIC_HEADER + "\n" + code) if inject_header else code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name

    try:
        cmd = [arg.replace("{file}", tmp_path) for arg in cmd_template]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            shell=False,
        )
        output = res.stdout + "\n" + res.stderr
        return {
            "success": True,
            "returncode": res.returncode,
            "raw_output": output,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "raw_output": "",
            "error": f"Timeout after {timeout_sec}s",
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "returncode": -1,
            "raw_output": "",
            "error": f"Executable not found: {e}",
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "raw_output": "",
            "error": str(e),
        }
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

def run_dslinter(code: str, inject_header: bool = True, timeout_sec: int = 15) -> Dict[str, Any]:
    """
    Executes dslinter (SERG-Delft) checking for data science iteration and DataFrame smells:
    - W5501: unassigned-dataframe
    - W5502: dataframe-iteration
    - W5503: dataframe-iteration-modification
    """
    python_exe = sys.executable
    cmd = [
        python_exe, "-m", "pylint",
        "--load-plugins=dslinter",
        "--disable=all",
        "--enable=W5501,W5502,W5503,dataframe-iteration,unassigned-dataframe",
        "--score=no",
        "{file}"
    ]

    exec_res = _run_command_on_code(code, cmd, inject_header=inject_header, timeout_sec=timeout_sec)
    if not exec_res["success"]:
        return {
            "available": False,
            "flagged": False,
            "codes": "",
            "warnings": [],
            "error": exec_res["error"],
        }

    raw = exec_res["raw_output"]
    warnings = []
    for line in raw.splitlines():
        if any(smell in line for smell in ["W5501", "W5502", "W5503", "dataframe-iteration", "unassigned-dataframe"]):
            warnings.append(line.strip())

    detected_codes = sorted(list(set(re.findall(r"W550\d|R550\d", raw))))

    return {
        "available": True,
        "flagged": len(warnings) > 0,
        "codes": ",".join(detected_codes),
        "warnings": warnings,
        "warning_count": len(warnings),
        "raw_output": raw,
    }

def run_perflint(code: str, inject_header: bool = True, timeout_sec: int = 15) -> Dict[str, Any]:
    """
    Executes perflint checking for Python performance anti-patterns:
    - W8201: loop-invariant-statement
    - W8202: loop-invariant-global-usage
    - W8203: dict-loop-variable
    - W8204: loop-invariant-branch
    - W8205: wasteful-comprehension (list comp in sum/any/all)
    """
    python_exe = sys.executable
    cmd = [
        python_exe, "-m", "pylint",
        "--load-plugins=perflint",
        "--disable=all",
        "--enable=W8201,W8202,W8203,W8204,W8205,loop-invariant-statement,wasteful-comprehension",
        "--score=no",
        "{file}"
    ]

    exec_res = _run_command_on_code(code, cmd, inject_header=inject_header, timeout_sec=timeout_sec)
    if not exec_res["success"]:
        return {
            "available": False,
            "flagged": False,
            "codes": "",
            "warnings": [],
            "error": exec_res["error"],
        }

    raw = exec_res["raw_output"]
    warnings = []
    for line in raw.splitlines():
        if any(code_id in line for code_id in ["W8201", "W8202", "W8203", "W8204", "W8205", "loop-invariant", "wasteful-comprehension"]):
            warnings.append(line.strip())

    detected_codes = sorted(list(set(re.findall(r"W820\d", raw))))

    return {
        "available": True,
        "flagged": len(warnings) > 0,
        "codes": ",".join(detected_codes),
        "warnings": warnings,
        "warning_count": len(warnings),
        "raw_output": raw,
    }

def run_pandas_vet(code: str, inject_header: bool = True, timeout_sec: int = 15) -> Dict[str, Any]:
    """
    Executes pandas-vet (Flake8 plugin) checking for Pandas API conventions (PD codes).
    """
    python_exe = sys.executable
    cmd = [
        python_exe, "-m", "flake8",
        "--select=PD",
        "{file}"
    ]

    exec_res = _run_command_on_code(code, cmd, inject_header=inject_header, timeout_sec=timeout_sec)
    if not exec_res["success"]:
        return {
            "available": False,
            "flagged": False,
            "codes": "",
            "warnings": [],
            "error": exec_res["error"],
        }

    raw = exec_res["raw_output"]
    warnings = [line.strip() for line in raw.splitlines() if ": PD" in line]
    detected_codes = sorted(list(set(re.findall(r"PD\d{3}", raw))))

    return {
        "available": True,
        "flagged": len(warnings) > 0,
        "codes": ",".join(detected_codes),
        "warnings": warnings,
        "warning_count": len(warnings),
        "raw_output": raw,
    }

def run_all_baselines(code: str, inject_header: bool = True, timeout_sec: int = 15) -> Dict[str, Any]:
    """
    Runs all 3 baseline linters (dslinter, perflint, pandas-vet) on a code snippet.
    """
    return {
        "dslinter": run_dslinter(code, inject_header=inject_header, timeout_sec=timeout_sec),
        "perflint": run_perflint(code, inject_header=inject_header, timeout_sec=timeout_sec),
        "pandas_vet": run_pandas_vet(code, inject_header=inject_header, timeout_sec=timeout_sec),
    }
