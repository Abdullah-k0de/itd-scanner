"""
Phase D: Master Study Pipeline Orchestrator.
Processes paired DS-1000 solutions (Score >= 4) and human reference code,
computing ITD smells, Radon complexity, Halstead metrics, and baseline linter flags.
Features:
- Exact rule codes captured for all linters (e.g. W8202, PD901, W5502)
- Line numbers and findings captured for ITD smells
- Incremental per-row saving (never lose progress)
- Automatic resumption from last checkpoint
- Reference solution metric caching (2-4x speedup)
"""

import os
import sys
import argparse
import json
import pandas as pd
from typing import Dict, Any, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from itd_scanner import scan_code, compute_code_metrics
from itd_scanner.baselines import run_all_baselines

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(BASE_DIR, "datasets", "phase2_score_4_plus.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "datasets", "phase2_final_metrics.csv")

REF_CACHE: Dict[Tuple[Any, bool], Dict[str, Any]] = {}

def analyze_reference(problem_id: Any, ref_code: str, run_baselines: bool) -> Dict[str, Any]:
    """Caches reference metrics by problem_id."""
    cache_key = (problem_id, run_baselines)
    if cache_key in REF_CACHE:
        return REF_CACHE[cache_key]

    ref_details = scan_code(ref_code, return_details=True)
    ref_itd = ref_details["flags"]
    ref_metrics = compute_code_metrics(ref_code)

    ref_findings_str = "; ".join(f"L{f['line']}: {f['smell']}" for f in ref_details["findings"])

    if run_baselines:
        ref_baselines = run_all_baselines(ref_code)
        ref_dslinter = ref_baselines["dslinter"]["flagged"]
        ref_dslinter_codes = ref_baselines["dslinter"].get("codes", "")
        ref_perflint = ref_baselines["perflint"]["flagged"]
        ref_perflint_codes = ref_baselines["perflint"].get("codes", "")
        ref_pandas_vet = ref_baselines["pandas_vet"]["flagged"]
        ref_pandas_vet_codes = ref_baselines["pandas_vet"].get("codes", "")
    else:
        ref_dslinter = False
        ref_dslinter_codes = ""
        ref_perflint = False
        ref_perflint_codes = ""
        ref_pandas_vet = False
        ref_pandas_vet_codes = ""

    cached_data = {
        "itd": ref_itd,
        "findings_str": ref_findings_str,
        "metrics": ref_metrics,
        "has_itd": any(ref_itd.values()),
        "itd_count": sum(1 for v in ref_itd.values() if v),
        "dslinter": ref_dslinter,
        "dslinter_codes": ref_dslinter_codes,
        "perflint": ref_perflint,
        "perflint_codes": ref_perflint_codes,
        "pandas_vet": ref_pandas_vet,
        "pandas_vet_codes": ref_pandas_vet_codes,
    }
    REF_CACHE[cache_key] = cached_data
    return cached_data

def process_single_row(row: pd.Series, run_baselines: bool = True) -> Dict[str, Any]:
    """
    Analyzes a single paired row:
    1. Scans AI-generated code (ITD smells, Radon metrics, Baselines with exact codes)
    2. Scans Complete Reference code (cached)
    3. Calculates comparative deltas (Delta CC, Delta Volume, ITD Introduced)
    """
    gen_code = str(row.get("generated_code", ""))
    ref_code = str(row.get("complete_reference_solution", ""))
    prob_id = row["problem_id"]

    # --- 1. Analyze AI Generated Code ---
    gen_details = scan_code(gen_code, return_details=True)
    gen_itd = gen_details["flags"]
    gen_metrics = compute_code_metrics(gen_code)

    gen_has_itd = any(gen_itd.values())
    gen_itd_count = sum(1 for v in gen_itd.values() if v)
    gen_findings_str = "; ".join(f"L{f['line']}: {f['smell']}" for f in gen_details["findings"])

    if run_baselines:
        gen_baselines = run_all_baselines(gen_code)
        gen_dslinter = gen_baselines["dslinter"]["flagged"]
        gen_dslinter_codes = gen_baselines["dslinter"].get("codes", "")
        gen_perflint = gen_baselines["perflint"]["flagged"]
        gen_perflint_codes = gen_baselines["perflint"].get("codes", "")
        gen_pandas_vet = gen_baselines["pandas_vet"]["flagged"]
        gen_pandas_vet_codes = gen_baselines["pandas_vet"].get("codes", "")
    else:
        gen_dslinter = False
        gen_dslinter_codes = ""
        gen_perflint = False
        gen_perflint_codes = ""
        gen_pandas_vet = False
        gen_pandas_vet_codes = ""

    # --- 2. Analyze Human Reference Code (Cached) ---
    ref_data = analyze_reference(prob_id, ref_code, run_baselines)
    ref_itd = ref_data["itd"]
    ref_metrics = ref_data["metrics"]

    # --- 3. Compute Comparative Deltas ---
    delta_cc = gen_metrics["cyclomatic_complexity"] - ref_metrics["cyclomatic_complexity"]
    delta_sloc = gen_metrics["sloc"] - ref_metrics["sloc"]
    delta_vol = round(gen_metrics["halstead_volume"] - ref_metrics["halstead_volume"], 2)
    delta_itd = gen_itd_count - ref_data["itd_count"]
    itd_introduced = gen_has_itd and not ref_data["has_itd"]

    # --- 4. Flattened Unified Schema ---
    return {
        # Metadata
        "problem_id": prob_id,
        "library": row["library"],
        "perturbation_type": row.get("perturbation_type", "Origin"),
        "model_config": row["model_config"],
        "functional_score": row["functional_score"],

        # AI Generated Code - ITD Flags
        "gen_collection_choker": gen_itd["Collection Choker"],
        "gen_math_looper": gen_itd["Math Looper"],
        "gen_string_masher": gen_itd["String Masher"],
        "gen_ram_hog": gen_itd["RAM Hog"],
        "gen_diy_wheel": gen_itd["DIY Wheel"],
        "gen_itd_count": gen_itd_count,
        "gen_has_itd": gen_has_itd,
        "gen_itd_findings": gen_findings_str,

        # AI Generated Code - Radon Metrics
        "gen_sloc": gen_metrics["sloc"],
        "gen_cyclomatic_complexity": gen_metrics["cyclomatic_complexity"],
        "gen_maintainability_index": gen_metrics["maintainability_index"],
        "gen_halstead_volume": gen_metrics["halstead_volume"],
        "gen_halstead_difficulty": gen_metrics["halstead_difficulty"],

        # AI Generated Code - Baseline Linters (Flags & Exact Codes)
        "gen_dslinter_flagged": gen_dslinter,
        "gen_dslinter_codes": gen_dslinter_codes,
        "gen_perflint_flagged": gen_perflint,
        "gen_perflint_codes": gen_perflint_codes,
        "gen_pandas_vet_flagged": gen_pandas_vet,
        "gen_pandas_vet_codes": gen_pandas_vet_codes,

        # Reference Code - ITD Flags
        "ref_collection_choker": ref_itd["Collection Choker"],
        "ref_math_looper": ref_itd["Math Looper"],
        "ref_string_masher": ref_itd["String Masher"],
        "ref_ram_hog": ref_itd["RAM Hog"],
        "ref_diy_wheel": ref_itd["DIY Wheel"],
        "ref_itd_count": ref_data["itd_count"],
        "ref_has_itd": ref_data["has_itd"],
        "ref_itd_findings": ref_data["findings_str"],

        # Reference Code - Radon Metrics
        "ref_sloc": ref_metrics["sloc"],
        "ref_cyclomatic_complexity": ref_metrics["cyclomatic_complexity"],
        "ref_maintainability_index": ref_metrics["maintainability_index"],
        "ref_halstead_volume": ref_metrics["halstead_volume"],
        "ref_halstead_difficulty": ref_metrics["halstead_difficulty"],

        # Reference Code - Baseline Linters (Flags & Exact Codes)
        "ref_dslinter_flagged": ref_data["dslinter"],
        "ref_dslinter_codes": ref_data["dslinter_codes"],
        "ref_perflint_flagged": ref_data["perflint"],
        "ref_perflint_codes": ref_data["perflint_codes"],
        "ref_pandas_vet_flagged": ref_data["pandas_vet"],
        "ref_pandas_vet_codes": ref_data["pandas_vet_codes"],

        # Comparative Deltas (AI vs Reference)
        "delta_cyclomatic_complexity": delta_cc,
        "delta_sloc": delta_sloc,
        "delta_halstead_volume": delta_vol,
        "delta_itd_count": delta_itd,
        "itd_introduced": itd_introduced,

        # Raw Code Snippets
        "generated_code": gen_code,
        "complete_reference_solution": ref_code,
    }

def main():
    parser = argparse.ArgumentParser(description="Run Phase D static analysis pipeline on DS-1000 solutions")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input paired dataset CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output master metrics CSV")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows")
    parser.add_argument("--no-baselines", action="store_true", help="Skip external baseline linters to run faster")
    parser.add_argument("--preview", action="store_true", help="Print detailed preview to console without saving")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading paired dataset from: {args.input}")
    df = pd.read_csv(args.input)
    total_input_rows = len(df)
    print(f"Total paired rows in input: {total_input_rows}")

    processed_keys = set()
    output_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    if output_exists:
        try:
            existing_df = pd.read_csv(args.output, usecols=["problem_id", "model_config"])
            for _, r in existing_df.iterrows():
                processed_keys.add((str(r["problem_id"]), str(r["model_config"])))
            print(f"Resuming: Found {len(processed_keys)} already processed records in '{args.output}'.")
        except Exception as e:
            print(f"Warning: Could not read existing output file ({e}). Starting fresh.")
            output_exists = False

    remaining_rows = []
    for _, row in df.iterrows():
        key = (str(row["problem_id"]), str(row["model_config"]))
        if key not in processed_keys:
            remaining_rows.append(row)

    print(f"Rows remaining to process: {len(remaining_rows)}")
    if not remaining_rows:
        print("All rows are already analyzed! Nothing to do.")
        return

    if args.limit:
        remaining_rows = remaining_rows[:args.limit]
        print(f"Limit applied: Processing next {len(remaining_rows)} rows.")

    if args.preview:
        sample_row = remaining_rows[0]
        print(f"\n{'='*70}")
        print(f" PREVIEW: Problem ID {sample_row['problem_id']} ({sample_row['model_config']})")
        print(f"{'='*70}")
        res = process_single_row(sample_row, run_baselines=not args.no_baselines)
        preview_data = {k: v for k, v in res.items() if k not in ("generated_code", "complete_reference_solution")}
        print(json.dumps(preview_data, indent=2))
        return

    print(f"\nStarting analysis... Results will be appended to '{args.output}' after each row.")
    for idx, row in enumerate(remaining_rows):
        res = process_single_row(row, run_baselines=not args.no_baselines)

        single_df = pd.DataFrame([res])
        single_df.to_csv(
            args.output,
            mode="a",
            header=not output_exists,
            index=False
        )
        output_exists = True

        prob_id = row['problem_id']
        cfg = row['model_config']
        itd_status = "[ITD DETECTED]" if res['gen_has_itd'] else "[CLEAN]"
        codes = []
        if res.get('gen_perflint_codes'): codes.append(f"Perflint:{res['gen_perflint_codes']}")
        if res.get('gen_pandas_vet_codes'): codes.append(f"PD:{res['gen_pandas_vet_codes']}")
        codes_str = f" ({', '.join(codes)})" if codes else ""

        print(f"[{idx + 1}/{len(remaining_rows)}] Problem {prob_id} ({cfg}): {itd_status}{codes_str} | Saved.")

    print(f"\nBatch Complete! All processed results are saved in '{args.output}'.")

if __name__ == "__main__":
    main()
