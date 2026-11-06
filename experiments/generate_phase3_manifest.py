"""
Phase 3 - Step 1 (Phase A): Candidate Selection & Pairing Manifest Generator.

Queries `datasets/phase2_final_metrics.csv` to isolate functionally verified (score >= 4)
candidate pairs where the LLM introduced Idiomatic Technical Debt (`itd_introduced == True`,
meaning `gen_has_itd == True` and `ref_has_itd == False`), joins with the DS-1000 harness
context from `datasets/ds1000_generated_edited.csv`, and outputs `datasets/phase3_candidates.csv`.
"""

from pathlib import Path
import re
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
PHASE2_METRICS_CSV = ROOT_DIR / "datasets" / "phase2_final_metrics.csv"
RAW_DS1000_CSV = ROOT_DIR / "datasets" / "ds1000_generated_edited.csv"
OUTPUT_CANDIDATES_CSV = ROOT_DIR / "datasets" / "phase3_candidates.csv"

# Compute- and memory-sensitive libraries targeted for Phase 3 input scaling
TARGET_LIBRARIES = {"pandas", "numpy", "pytorch", "scipy", "sklearn", "tensorflow"}


def determine_primary_smell(row: pd.Series) -> str:
    """Assign a primary ITD category for stratified profiling and plotting."""
    if row.get("gen_collection_choker", 0) > 0:
        return "Collection Choker"
    if row.get("gen_math_looper", 0) > 0:
        return "Math Looper"
    if row.get("gen_diy_wheel", 0) > 0:
        return "DIY Wheel"
    if row.get("gen_string_masher", 0) > 0:
        return "String Masher"
    if row.get("gen_ram_hog", 0) > 0:
        return "RAM Hog"
    return "None"


def list_all_smells(row: pd.Series) -> str:
    """Return a semicolon-delimited list of all active ITD smells in the generated snippet."""
    smells = []
    if row.get("gen_collection_choker", 0) > 0:
        smells.append("Collection Choker")
    if row.get("gen_math_looper", 0) > 0:
        smells.append("Math Looper")
    if row.get("gen_diy_wheel", 0) > 0:
        smells.append("DIY Wheel")
    if row.get("gen_string_masher", 0) > 0:
        smells.append("String Masher")
    if row.get("gen_ram_hog", 0) > 0:
        smells.append("RAM Hog")
    return "; ".join(smells)


def uses_unmocked_disk_io(code: str) -> bool:
    """Check whether code attempts to read a non-existent local file via open(...) or read_csv('...')."""
    if not isinstance(code, str):
        return False
    # Check for open('something.csv') or pd.read_csv('something.csv') without StringIO
    if re.search(r"\bopen\s*\(\s*['\"][^'\"]+['\"]", code):
        return True
    if re.search(r"\bread_csv\s*\(\s*['\"][^'\"]+['\"]", code) and "StringIO" not in code:
        return True
    return False


def generate_manifest() -> pd.DataFrame:
    print(f"Loading Phase 2 metrics from: {PHASE2_METRICS_CSV}")
    df_p2 = pd.read_csv(PHASE2_METRICS_CSV)

    print(f"Loading raw DS-1000 harness context from: {RAW_DS1000_CSV}")
    df_raw = pd.read_csv(
        RAW_DS1000_CSV,
        usecols=[
            "problem_id",
            "library_problem_id",
            "test_case_cnt",
            "prompt",
            "code_context",
            "reference_code",
        ],
    )

    # Filter criteria per Phase 3A Blueprint:
    # 1. Functionally valid (functional_score >= 4)
    # 2. Architectural contrast: AI has ITD (gen_has_itd == True) while Reference is clean (ref_has_itd == False)
    # 3. Compute/memory-sensitive libraries (exclude Matplotlib plotting tasks)
    mask_score = df_p2["functional_score"] >= 4
    mask_itd_contrast = (df_p2["itd_introduced"] == True) & (df_p2["gen_has_itd"] == True) & (df_p2["ref_has_itd"] == False)
    mask_library = df_p2["library"].astype(str).str.lower().isin(TARGET_LIBRARIES)

    candidates = df_p2[mask_score & mask_itd_contrast & mask_library].copy()

    # Merge DS-1000 harness columns
    candidates = candidates.merge(df_raw, on="problem_id", how="left")

    # Add Phase 3 stratification & diagnostic columns
    candidates["primary_smell"] = candidates.apply(determine_primary_smell, axis=1)
    candidates["active_smells"] = candidates.apply(list_all_smells, axis=1)
    candidates["is_core_library"] = candidates["library"].astype(str).str.lower().isin({"pandas", "numpy", "pytorch"})
    candidates["requires_disk_mock"] = candidates["generated_code"].apply(uses_unmocked_disk_io)

    # Sort cleanly: Control group first, then Core libraries (Pandas/NumPy/PyTorch), then by problem_id
    candidates["is_control_group"] = candidates["model_config"].isin(["27B_Control", "9B_Control"])
    candidates = candidates.sort_values(
        by=["is_control_group", "is_core_library", "problem_id", "model_config"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    # Assign a sequential candidate_id for easy indexing
    candidates.insert(0, "candidate_id", range(1, len(candidates) + 1))

    OUTPUT_CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUTPUT_CANDIDATES_CSV, index=False)

    print(f"\nSaved {len(candidates)} paired candidates to: {OUTPUT_CANDIDATES_CSV}")
    print("\n--- Breakdown by Library & Model Config ---")
    print(candidates.groupby(["library", "model_config"]).size().unstack(fill_value=0))

    print("\n--- Breakdown by Primary ITD Smell ---")
    print(candidates.groupby(["primary_smell", "model_config"]).size().unstack(fill_value=0))

    print("\n--- Core vs. Extended Libraries ---")
    print(f"Core Libraries (Pandas, NumPy, PyTorch): {candidates['is_core_library'].sum()}")
    print(f"Extended Libraries (SciPy, Sklearn, TensorFlow): {(~candidates['is_core_library']).sum()}")
    print(f"Snippets requiring virtual file mock (e.g. open('data.csv')): {candidates['requires_disk_mock'].sum()}")

    return candidates


if __name__ == "__main__":
    generate_manifest()

