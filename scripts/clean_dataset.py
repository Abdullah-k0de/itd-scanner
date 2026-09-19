import os
import shutil
import pandas as pd

# Support running from either root or scripts/ directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.getcwd()) == "scripts" else os.getcwd()
INPUT_FILE = os.path.join(BASE_DIR, "datasets", "ds1000_generated_edited.csv")
BACKUP_FILE = os.path.join(BASE_DIR, "datasets", "ds1000_generated_edited_backup.csv")

def clean_empty_rows():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File '{INPUT_FILE}' not found.")
        return

    # Create a backup first to preserve data
    if not os.path.exists(BACKUP_FILE):
        print(f"Creating backup at: {BACKUP_FILE}")
        shutil.copyfile(INPUT_FILE, BACKUP_FILE)
    else:
        print(f"Backup already exists at: {BACKUP_FILE}")

    print(f"Reading '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    initial_count = len(df)
    print(f"Total rows before cleaning: {initial_count}")

    # 1. Remove rows where all columns are NaN / empty
    df_cleaned = df.dropna(how="all")
    all_nan_dropped = initial_count - len(df_cleaned)

    # 2. Remove rows where problem_id is NaN or whitespace (invalid/empty records)
    if "problem_id" in df_cleaned.columns:
        valid_mask = df_cleaned["problem_id"].notna() & (df_cleaned["problem_id"].astype(str).str.strip() != "")
        empty_id_dropped = len(df_cleaned) - valid_mask.sum()
        df_cleaned = df_cleaned[valid_mask]
    else:
        empty_id_dropped = 0

    final_count = len(df_cleaned)
    total_dropped = initial_count - final_count

    print(f"\n--- Cleaning Summary ---")
    print(f"Completely empty rows dropped: {all_nan_dropped}")
    print(f"Rows with missing/empty problem_id dropped: {empty_id_dropped}")
    print(f"Total rows dropped: {total_dropped}")
    print(f"Rows remaining: {final_count}")

    if total_dropped > 0:
        print(f"\nSaving cleaned data back to '{INPUT_FILE}'...")
        df_cleaned.to_csv(INPUT_FILE, index=False)
        print("Done! File updated successfully.")
    else:
        print("\nNo empty rows found based on the current criteria.")

if __name__ == "__main__":
    clean_empty_rows()
