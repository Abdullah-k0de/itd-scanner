"""
Command-line interface for itd-scanner.
"""

import argparse
import sys
import json
from itd_scanner import scan_file

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(
        description="itd-scanner: Static Analysis for Idiomatic Technical Debt in Data Science Python"
    )
    parser.add_argument("file", help="Python source file to analyze")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--metrics", action="store_true", help="Include Radon software metrics (CC, LOC, MI)")

    args = parser.parse_args()

    try:
        report = scan_file(args.file, return_details=True, include_metrics=args.metrics)
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f" ITD-Scanner Report: {args.file}")
        print(f"{'='*60}")

        if report.get("syntax_error"):
            print("[!] Warning: File contains SyntaxErrors; static parse was incomplete.\n")

        print("Summary of Anti-Patterns:")
        for smell, detected in report["flags"].items():
            status = "[DETECTED]" if detected else "[CLEAN]"
            print(f"  {smell:<20}: {status}")

        if report["findings"]:
            print(f"\nDetailed Findings ({len(report['findings'])}):")
            for f in report["findings"]:
                print(f"  - Line {f['line']:<3} [{f['smell']}]: {f['message']}")
        else:
            print("\nNo idiomatic technical debt detected.")

        if args.metrics and "metrics" in report:
            m = report["metrics"]
            print(f"\nCode Metrics (Radon):")
            print(f"  - LOC (Lines of Code)      : {m.get('loc')}")
            print(f"  - SLOC (Source Lines)      : {m.get('sloc')}")
            print(f"  - Cyclomatic Complexity    : {m.get('cyclomatic_complexity')}")
            print(f"  - Maintainability Index    : {m.get('maintainability_index')}")

        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
