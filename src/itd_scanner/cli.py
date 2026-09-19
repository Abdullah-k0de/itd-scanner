"""
Command-line interface for itd-scanner.
"""

import argparse
import sys
import json
from itd_scanner import scan_code

def main():
    parser = argparse.ArgumentParser(
        description="itd-scanner: Static Analysis for Idiomatic Technical Debt in Data Science Python"
    )
    parser.add_argument("file", help="Python source file to analyze")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    results = scan_code(code)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n--- itd-scanner Results for: {args.file} ---")
        for smell, detected in results.items():
            status = "🚨 DETECTED" if detected else "✅ Clean"
            print(f"  {smell:<20}: {status}")

if __name__ == "__main__":
    main()
