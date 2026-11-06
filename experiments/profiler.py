"""
Phase 3 - Step 3 (Phase C): Isolated Hardware Profiling Engine (`profiler.py`).

Implements sandboxed worker execution (`multiprocessing.Process`) to measure:
1. CPU Execution Latency (`time.perf_counter()`) in milliseconds:
   - 1 warm-up execution (discarded to eliminate cold-start caching artifacts)
   - Up to 5 timed repetitions (`repetitions=5`)
   - Computes both trimmed mean (dropping min and max) and median latency (ms)
2. Peak Heap Memory Allocation (`tracemalloc`) in megabytes (MB):
   - Initialized via `tracemalloc.start()` / `tracemalloc.reset_peak()` strictly around
     the algorithm execution block (`exec_code`), excluding imports and synthetic input creation.
3. Hard Resource Ceilings:
   - Enforces a strict execution timeout (default 60s) with automatic process termination
     (`process.terminate()`) to catch unoptimized polynomial loops (`O(n^2)`) without hanging.
   - Restores fresh copies of scaled inputs before each repetition so in-place mutations
     (such as `df.pop(...)` or `df.replace(..., inplace=True)`) never corrupt subsequent runs.
"""

from __future__ import annotations

import builtins
import copy
import gc
import io
import math
import multiprocessing as mp
import statistics
import sys
import time
import traceback
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from scaler import (
    TIER_SIZES,
    inject_load_data_hook,
    normalize_and_split_script,
    scale_environment,
)


@dataclass
class ProfileResult:
    """Hardware execution metrics for a single (script, tier) execution."""
    status: str                  # "SUCCESS", "TIMEOUT", "MEMORY_ERROR", "RUNTIME_ERROR"
    tier: int                    # 1, 2, or 3
    target_n: int                # 1_000, 100_000, or 1_000_000
    latency_trimmed_mean_ms: float
    latency_median_ms: float
    latency_min_ms: float
    latency_max_ms: float
    peak_ram_mb: float
    num_runs: int
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clone_scaled_env(base_env: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a fast, isolated copy of the scaled execution environment before each
    timed repetition so in-place mutations (e.g. `df.pop()`, `inplace=True`, `arr[:] = ...`)
    do not corrupt subsequent repetitions.
    """
    run_env: Dict[str, Any] = {}
    for k, v in base_env.items():
        if k.startswith("__") or callable(v) or isinstance(v, type):
            run_env[k] = v
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            run_env[k] = v.copy(deep=True)
        elif isinstance(v, np.ndarray):
            run_env[k] = v.copy()
        elif HAS_TORCH and isinstance(v, torch.Tensor):
            run_env[k] = v.clone()
        elif isinstance(v, (list, dict, set)):
            run_env[k] = copy.deepcopy(v)
        elif isinstance(v, tuple):
            run_env[k] = copy.deepcopy(v)
        else:
            run_env[k] = v
    return run_env


def _worker_profile_script(
    code: str,
    code_context: Optional[str],
    tier: int,
    target_n: int,
    repetitions: int,
    timeout_sec: float,
    result_queue: mp.Queue,
) -> None:
    """
    Sandboxed worker function executed inside an isolated `multiprocessing.Process`.
    """
    # Prevent any Matplotlib window popups or cache warnings if imported by the script
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    os.environ.setdefault(" MPLBACKEND", "Agg")
    os.environ["MPLBACKEND"] = "Agg"

    # Silence stdout and print() during execution so console I/O is never timed
    orig_stdout = sys.stdout
    orig_print = builtins.print
    sys.stdout = io.StringIO()
    builtins.print = lambda *args, **kwargs: None

    try:
        setup_code, exec_code = normalize_and_split_script(code)
        compiled_setup = compile(setup_code, "<setup_code>", "exec")
        compiled_exec = compile(exec_code, "<exec_code>", "exec")

        base_env: Dict[str, Any] = {"__builtins__": builtins, "__name__": "__main__"}
        inject_load_data_hook(base_env, code_context)

        # Phase 1: Execute Setup & Scale Input Environment (Untimed)
        exec(compiled_setup, base_env)
        scale_environment(base_env, target_n=target_n)

        # Phase 2: Warm-up Run (1 execution to clear cold-start cache & measure peak RAM)
        warmup_env = _clone_scaled_env(base_env)
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()

        t_warm_start = time.perf_counter()
        exec(compiled_exec, warmup_env)
        t_warm_elapsed = time.perf_counter() - t_warm_start

        _, peak_bytes_warmup = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del warmup_env
        gc.collect()

        # Adaptive repetition guard: if a single run takes a big fraction of timeout_sec,
        # reduce repetitions so the worker finishes cleanly before the hard kill timeout.
        remaining_budget = max(0.0, timeout_sec - t_warm_elapsed - 2.0)
        if t_warm_elapsed > 0:
            max_possible_reps = int(remaining_budget // max(t_warm_elapsed, 0.001))
            actual_reps = max(1, min(repetitions, max_possible_reps))
        else:
            actual_reps = repetitions

        timings_ms: List[float] = []
        peak_bytes_max: int = peak_bytes_warmup

        # Phase 3: Timed Repetitions
        for rep in range(actual_reps):
            run_env = _clone_scaled_env(base_env)
            gc.collect()

            # Track peak heap on the first timed repetition as well
            track_mem = (rep == 0)
            if track_mem:
                tracemalloc.start()
                tracemalloc.reset_peak()

            t0 = time.perf_counter()
            exec(compiled_exec, run_env)
            t1 = time.perf_counter()

            if track_mem:
                _, rep_peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                if rep_peak_bytes > peak_bytes_max:
                    peak_bytes_max = rep_peak_bytes

            timings_ms.append((t1 - t0) * 1000.0)
            del run_env

        # Compute trimmed mean (drop min and max when >= 3 runs) and median
        sorted_t = sorted(timings_ms)
        if len(sorted_t) >= 3:
            trimmed = sorted_t[1:-1]
            trimmed_mean_ms = float(sum(trimmed) / len(trimmed))
        else:
            trimmed_mean_ms = float(sum(sorted_t) / len(sorted_t))

        median_ms = float(statistics.median(sorted_t))
        min_ms = float(sorted_t[0])
        max_ms = float(sorted_t[-1])
        peak_ram_mb = float(peak_bytes_max) / (1024.0 * 1024.0)

        result_queue.put(
            ProfileResult(
                status="SUCCESS",
                tier=tier,
                target_n=target_n,
                latency_trimmed_mean_ms=round(trimmed_mean_ms, 4),
                latency_median_ms=round(median_ms, 4),
                latency_min_ms=round(min_ms, 4),
                latency_max_ms=round(max_ms, 4),
                peak_ram_mb=round(peak_ram_mb, 4),
                num_runs=len(timings_ms),
                error_message="",
            ).to_dict()
        )

    except MemoryError as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        result_queue.put(
            ProfileResult(
                status="MEMORY_ERROR",
                tier=tier,
                target_n=target_n,
                latency_trimmed_mean_ms=float("nan"),
                latency_median_ms=float("nan"),
                latency_min_ms=float("nan"),
                latency_max_ms=float("nan"),
                peak_ram_mb=float("nan"),
                num_runs=0,
                error_message=f"MemoryError: {exc}",
            ).to_dict()
        )
    except Exception as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tb_short = traceback.format_exc().splitlines()[-1]
        result_queue.put(
            ProfileResult(
                status="RUNTIME_ERROR",
                tier=tier,
                target_n=target_n,
                latency_trimmed_mean_ms=float("nan"),
                latency_median_ms=float("nan"),
                latency_min_ms=float("nan"),
                latency_max_ms=float("nan"),
                peak_ram_mb=float("nan"),
                num_runs=0,
                error_message=tb_short,
            ).to_dict()
        )
    finally:
        sys.stdout = orig_stdout
        builtins.print = orig_print


def profile_script_isolated(
    code: str,
    code_context: Optional[str] = None,
    tier: int = 1,
    target_n: Optional[int] = None,
    repetitions: int = 5,
    timeout_sec: float = 60.0,
) -> ProfileResult:
    """
    Execute and profile `code` at `tier` (`target_n` elements) inside a sandboxed
    worker process with a hard `timeout_sec` ceiling.
    """
    if target_n is None:
        target_n = TIER_SIZES.get(tier, 1_000)

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker_profile_script,
        args=(code, code_context, tier, target_n, repetitions, timeout_sec, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        return ProfileResult(
            status="TIMEOUT",
            tier=tier,
            target_n=target_n,
            latency_trimmed_mean_ms=round(timeout_sec * 1000.0, 4),
            latency_median_ms=round(timeout_sec * 1000.0, 4),
            latency_min_ms=round(timeout_sec * 1000.0, 4),
            latency_max_ms=round(timeout_sec * 1000.0, 4),
            peak_ram_mb=float("nan"),
            num_runs=0,
            error_message=f"Exceeded hard timeout of {timeout_sec}s",
        )

    if not result_queue.empty():
        data = result_queue.get()
        return ProfileResult(**data)

    return ProfileResult(
        status="RUNTIME_ERROR",
        tier=tier,
        target_n=target_n,
        latency_trimmed_mean_ms=float("nan"),
        latency_median_ms=float("nan"),
        latency_min_ms=float("nan"),
        latency_max_ms=float("nan"),
        peak_ram_mb=float("nan"),
        num_runs=0,
        error_message=f"Worker process exited abnormally with code {proc.exitcode}",
    )


def profile_candidate_pair(
    candidate_row: pd.Series,
    tiers: List[int] = [1, 2, 3],
    repetitions: int = 5,
    timeout_sec: float = 60.0,
) -> List[Dict[str, Any]]:
    """
    Profile both the LLM-generated solution (`generated_code`) and the paired human
    reference solution (`complete_reference_solution`) across the requested scale tiers
    and compute hardware degradation ratios:
        delta_latency_ratio = Latency_LLM / Latency_Ref
        delta_memory_ratio  = PeakRAM_LLM / PeakRAM_Ref
    """
    records: List[Dict[str, Any]] = []
    gen_code = str(candidate_row["generated_code"])
    ref_code = str(candidate_row["complete_reference_solution"])
    code_context = candidate_row.get("code_context", None)

    gen_timed_out_prev_tier = False

    for tier in tiers:
        target_n = TIER_SIZES[tier]

        # 1. Profile Human Reference Solution (Control Group)
        ref_res = profile_script_isolated(
            code=ref_code,
            code_context=code_context,
            tier=tier,
            target_n=target_n,
            repetitions=repetitions,
            timeout_sec=timeout_sec,
        )

        # 2. Profile AI-Generated Solution (Target Group)
        # If the AI solution already timed out at a smaller tier (e.g. O(N^2) at Tier 2),
        # skip running Tier 3 for 60s and record TIMEOUT_SKIPPED immediately.
        if gen_timed_out_prev_tier:
            gen_res = ProfileResult(
                status="TIMEOUT",
                tier=tier,
                target_n=target_n,
                latency_trimmed_mean_ms=round(timeout_sec * 1000.0, 4),
                latency_median_ms=round(timeout_sec * 1000.0, 4),
                latency_min_ms=round(timeout_sec * 1000.0, 4),
                latency_max_ms=round(timeout_sec * 1000.0, 4),
                peak_ram_mb=float("nan"),
                num_runs=0,
                error_message=f"Skipped Tier {tier} after timing out on lower tier",
            )
        else:
            gen_res = profile_script_isolated(
                code=gen_code,
                code_context=code_context,
                tier=tier,
                target_n=target_n,
                repetitions=repetitions,
                timeout_sec=timeout_sec,
            )
            if gen_res.status == "TIMEOUT":
                gen_timed_out_prev_tier = True

        # 3. Compute Comparative Hardware Degradation Ratios (Phase 3D)
        if (
            ref_res.status == "SUCCESS"
            and gen_res.status in {"SUCCESS", "TIMEOUT"}
            and ref_res.latency_trimmed_mean_ms > 0
        ):
            delta_latency_ratio = round(
                gen_res.latency_trimmed_mean_ms / max(ref_res.latency_trimmed_mean_ms, 0.0001), 4
            )
        else:
            delta_latency_ratio = float("nan")

        if (
            ref_res.status == "SUCCESS"
            and gen_res.status == "SUCCESS"
            and not math.isnan(gen_res.peak_ram_mb)
        ):
            # Floor reference peak heap at 0.001 MB (1 KB) to avoid division by zero on in-place C ops
            delta_memory_ratio = round(
                max(gen_res.peak_ram_mb, 0.0001) / max(ref_res.peak_ram_mb, 0.001), 4
            )
        else:
            delta_memory_ratio = float("nan")

        records.append(
            {
                "candidate_id": int(candidate_row["candidate_id"]),
                "problem_id": int(candidate_row["problem_id"]),
                "library": candidate_row["library"],
                "model_config": candidate_row["model_config"],
                "functional_score": candidate_row["functional_score"],
                "primary_smell": candidate_row["primary_smell"],
                "active_smells": candidate_row.get("active_smells", ""),
                "gen_itd_count": int(candidate_row["gen_itd_count"]),
                "gen_cyclomatic_complexity": candidate_row["gen_cyclomatic_complexity"],
                "ref_cyclomatic_complexity": candidate_row["ref_cyclomatic_complexity"],
                "delta_cyclomatic_complexity": candidate_row["delta_cyclomatic_complexity"],
                "gen_sloc": int(candidate_row["gen_sloc"]),
                "ref_sloc": int(candidate_row["ref_sloc"]),
                "delta_sloc": int(candidate_row["delta_sloc"]),
                "tier": tier,
                "scale_n": target_n,
                "gen_status": gen_res.status,
                "ref_status": ref_res.status,
                "gen_latency_ms": gen_res.latency_trimmed_mean_ms,
                "ref_latency_ms": ref_res.latency_trimmed_mean_ms,
                "gen_latency_median_ms": gen_res.latency_median_ms,
                "ref_latency_median_ms": ref_res.latency_median_ms,
                "delta_latency_ratio": delta_latency_ratio,
                "gen_peak_ram_mb": gen_res.peak_ram_mb,
                "ref_peak_ram_mb": ref_res.peak_ram_mb,
                "delta_memory_ratio": delta_memory_ratio,
                "gen_num_runs": gen_res.num_runs,
                "ref_num_runs": ref_res.num_runs,
                "gen_error": gen_res.error_message,
                "ref_error": ref_res.error_message,
            }
        )

    return records

