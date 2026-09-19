import os
import re
import pandas as pd

# Support running from either root or scripts/ directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.getcwd()) == "scripts" else os.getcwd()
INPUT_PATH = os.path.join(BASE_DIR, "datasets", "ds1000_generated_edited.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "datasets", "phase2_score_4_plus.csv")

def clean_code(raw_code: str) -> str:
    """Strips markdown code fences and leading/trailing whitespace."""
    code = raw_code.strip()
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", code, re.DOTALL)
    if match:
        code = match.group(1).strip()
    return code

print(f"Loading dataset from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
valid_solutions = []

model_configs = [
    ("27B_Control", "generated_code_27b", "generated_code_27b_score"),
    ("27B_Negative", "generated_code_27b_negative", "generated_code_27b_negative_score"),
    ("9B_Control", "generated_code_9b", "generated_code_9b_score"),
    ("9B_Negative", "generated_code_9b_negative", "generated_code_9b_negative_score"),
]

for config_name, code_col, score_col in model_configs:
    numeric_scores = pd.to_numeric(df[score_col], errors="coerce")
    valid_rows = df[numeric_scores >= 4]

    for _, row in valid_rows.iterrows():
        raw_code = row[code_col]

        if pd.isna(raw_code):
            continue

        raw_code_str = str(raw_code).strip()
        if not raw_code_str or raw_code_str.startswith("ERROR:"):
            continue

        source_code = clean_code(raw_code_str)

        valid_solutions.append({
            "problem_id": row["problem_id"],
            "library": row["library"],
            "perturbation_type": row.get("perturbation_type", "Origin"),
            "model_config": config_name,
            "functional_score": row[score_col],
            "source_code": source_code,
        })

clean_df = pd.DataFrame(valid_solutions)
clean_df.to_csv(OUTPUT_PATH, index=False)
print(f"Extraction complete: {len(clean_df)} valid solutions extracted and saved to '{OUTPUT_PATH}'.")
