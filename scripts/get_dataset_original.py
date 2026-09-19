import os
import gzip
import json
import urllib.request
import pandas as pd

# Support running from either root or scripts/ directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.getcwd()) == "scripts" else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ds_1000_original.csv")
JSONL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ds_1000.jsonl")

CANDIDATE_URLS = [
    "https://github.com/xlang-ai/DS-1000/raw/main/data/ds1000.jsonl.gz",
    "https://huggingface.co/datasets/xlangai/DS-1000/resolve/main/data/ds1000.jsonl.gz",
    "https://huggingface.co/datasets/xlangai/DS-1000/resolve/main/test.jsonl",
]

def download_and_extract():
    headers = {"User-Agent": "Mozilla/5.0"}
    downloaded_bytes = None
    source_url = None

    print("Fetching DS-1000 dataset...")
    for url in CANDIDATE_URLS:
        try:
            print(f"Trying {url}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status == 200:
                    downloaded_bytes = resp.read()
                    source_url = url
                    print(f"Successfully downloaded from: {url} ({len(downloaded_bytes) / 1024 / 1024:.2f} MB)")
                    break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    if not downloaded_bytes:
        raise RuntimeError("Could not download DS-1000 from any candidate source.")

    if source_url.endswith(".gz") or downloaded_bytes[:2] == b"\x1f\x8b":
        print("Decompressing gzip data...")
        content = gzip.decompress(downloaded_bytes).decode("utf-8")
    else:
        content = downloaded_bytes.decode("utf-8")

    print(f"Saving raw JSONL to {JSONL_OUTPUT_PATH}...")
    with open(JSONL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print("Parsing records into pandas DataFrame...")
    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records. Columns: {list(df.columns)}")

    print(f"Saving DataFrame to {CSV_OUTPUT_PATH}...")
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print("Done! Dataset downloaded and saved successfully.")

if __name__ == "__main__":
    download_and_extract()
