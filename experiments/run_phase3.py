"""
Phase 3 - Step 4 (Phase D): Dynamic Hardware Profiling CLI Orchestrator (`run_phase3.py`).

Features:
- Auto-Resume by Default: Automatically checks `datasets/phase3_dynamic_profiles.csv` on
  startup, skips any `candidate_id` already completed, and appends new results immediately
  after every single candidate finishes. If interrupted (Ctrl+C or PC restart), running the
  exact same command resumes seamlessly from where it left off.
- Supports filtering by `--candidate-id`, `--problem-id`, `--library`, `--model-config`,
  and `--limit`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set
import pandas as pd

from profiler import profile_candidate_pair


ROOT_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_CSV = ROOT_DIR / "datasets" / "phase3_candidates.csv"
DEFAULT_OUTPUT_CSV = ROOT_DIR / "datasets" / "phase3_dynamic_profiles.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: Input Scaling & Dynamic Hardware Profiling Runner (Auto-Resumable)"
    )
    parser.add_argument(
        "--candidate-id",
        type=int,
        default=None,
        help="Run a single specific candidate_id from phase3_candidates.csv",
    )
    parser.add_argument(
        "--problem-id",
        type=int,
        default=None,
        help="Filter to a specific DS-1000 problem_id",
    )
    parser.add_argument(
        "--library",
        type=str,
        default=None,
        help="Filter to a specific library (e.g., Pandas, Numpy, Pytorch, Scipy, Sklearn)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Filter to a specific model_config (e.g., 27B_Control, 9B_Control, 27B_Negative, 9B_Negative)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of remaining candidate pairs to profile in this run",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="1,2,3",
        help="Comma-separated list of scaling tiers to run (default: '1,2,3')",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of timed repetitions per tier after warm-up (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Hard timeout in seconds per (script, tier) worker execution (default: 30.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output CSV path (default: datasets/phase3_dynamic_profiles.csv)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV from scratch instead of auto-resuming and appending",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(
            f"Candidate manifest not found at {CANDIDATES_CSV}. Run generate_phase3_manifest.py first."
        )

    df = pd.read_csv(CANDIDATES_CSV)
    # Exclude the 4 snippets requiring unmocked external disk files unless explicitly requested
    if args.candidate_id is None and args.problem_id is None:
        df = df[~df["requires_disk_mock"]].copy()

    if args.candidate_id is not None:
        df = df[df["candidate_id"] == args.candidate_id].copy()
    if args.problem_id is not None:
        df = df[df["problem_id"] == args.problem_id].copy()
    if args.library is not None:
        df = df[df["library"].astype(str).str.lower() == args.library.lower()].copy()
    if args.model_config is not None:
        df = df[df["model_config"].astype(str).str.lower() == args.model_config.lower()].copy()

    tiers: List[int] = [int(t.strip()) for t in args.tiers.split(",") if t.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-resume check: load already-completed candidate_ids unless --overwrite is passed
    completed_cids: Set[int] = set()
    if out_path.exists() and not args.overwrite:
        try:
            existing_df = pd.read_csv(out_path)
            if "candidate_id" in existing_df.columns and "tier" in existing_df.columns:
                # A candidate is considered complete if all requested tiers are already logged for it
                tier_counts = existing_df.groupby("candidate_id")["tier"].nunique()
                completed_cids = set(tier_counts[tier_counts >= len(tiers)].index.astype(int))
                # If a candidate was only partially written before an abrupt kill, remove its partial rows
                partial_cids = set(existing_df["candidate_id"].astype(int)) - completed_cids
                if partial_cids:
                    cleaned_df = existing_df[~existing_df["candidate_id"].astype(int).isin(partial_cids)]
                    cleaned_df.to_csv(out_path, index=False)
        except Exception as exc:
            print(f"[Warning] Could not read existing {out_path} for auto-resume: {exc}")
    elif out_path.exists() and args.overwrite:
        out_path.unlink()

    total_matching = len(df)
    if completed_cids:
        df = df[~df["candidate_id"].astype(int).isin(completed_cids)].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("=" * 92)
    print(
        f"PHASE 3 DYNAMIC HARDWARE PROFILER | Matching: {total_matching} | "
        f"Already Completed: {len(completed_cids)} | Remaining to Run: {len(df)}"
    )
    print(f"Tiers: {tiers} | Repetitions: {args.repetitions} (+1 warmup) | Timeout per tier: {args.timeout}s")
    print(f"Output CSV (Incremental Auto-Append): {out_path}")
    print("=" * 92)

    if df.empty:
        print("All matching candidates are already completed! Pass --overwrite to re-run from scratch.")
        return

    newly_saved_rows = 0
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        cid = int(row["candidate_id"])
        pid = int(row["problem_id"])
        lib = row["library"]
        cfg = row["model_config"]
        smell = row["primary_smell"]
        print(f"\n[{idx}/{len(df)}] Candidate #{cid} | Problem {pid} ({lib} | {cfg}) | Smell: {smell}")
        print("-" * 92)

        records = profile_candidate_pair(
            candidate_row=row,
            tiers=tiers,
            repetitions=args.repetitions,
            timeout_sec=args.timeout,
        )
        for rec in records:
            t = rec["tier"]
            n = rec["scale_n"]
            g_lat = rec["gen_latency_ms"]
            r_lat = rec["ref_latency_ms"]
            d_lat = rec["delta_latency_ratio"]
            g_ram = rec["gen_peak_ram_mb"]
            r_ram = rec["ref_peak_ram_mb"]
            d_ram = rec["delta_memory_ratio"]
            g_st = rec["gen_status"]
            r_st = rec["ref_status"]
            print(
                f"  Tier {t} (N={n:>9,d}) | "
                f"AI: {g_lat:>10.3f} ms ({g_st}) vs Ref: {r_lat:>8.3f} ms ({r_st}) -> Slowdown: {d_lat:>8.2f}x | "
                f"RAM AI: {g_ram:>7.3f} MB vs Ref: {r_ram:>7.3f} MB -> MemRatio: {d_ram:>7.2f}x"
            )
            if rec["gen_error"]:
                print(f"    [AI Note]: {rec['gen_error']}")
            if rec["ref_error"]:
                print(f"    [Ref Note]: {rec['ref_error']}")

        # Append immediately to CSV after this candidate completes
        rec_df = pd.DataFrame(records)
        write_header = not out_path.exists() or out_path.stat().st_size == 0
        rec_df.to_csv(out_path, mode="a", header=write_header, index=False)
        newly_saved_rows += len(records)

    print("\n" + "=" * 92)
    print(f"Run finished! Appended {newly_saved_rows} new tier rows to: {out_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
