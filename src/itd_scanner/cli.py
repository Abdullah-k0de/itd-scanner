"""
Command-line interface for itd-scanner.
Provides rich, color-coded terminal reports of Idiomatic Technical Debt (ITD)
and software complexity metrics in Data Science Python code.
"""

import argparse
import sys
import os
import json
from itd_scanner import scan_file

# Initialize VT100 terminal mode on Windows consoles for ANSI colors
if sys.platform == "win32":
    os.system("")

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GRAY = "\033[90m"

def format_banner(title: str) -> str:
    line = "=" * 65
    return f"\n{CYAN}{BOLD}{line}\n  {title}\n{line}{RESET}"

def main():
    parser = argparse.ArgumentParser(
        description="itd-scanner: Static Analysis for Idiomatic Technical Debt in Data Science Python"
    )
    parser.add_argument("file", help="Python source file to analyze")
    parser.add_argument("--json", action="store_true", help="Output results in raw JSON format")
    parser.add_argument("--metrics", action="store_true", help="Include Radon software metrics (CC, LOC, Halstead, MI)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color codes")

    args = parser.parse_args()

    # Disable colors if requested or redirected
    global RESET, BOLD, RED, GREEN, YELLOW, BLUE, CYAN, GRAY
    if args.no_color or not sys.stdout.isatty():
        RESET = BOLD = RED = GREEN = YELLOW = BLUE = CYAN = GRAY = ""

    try:
        report = scan_file(args.file, return_details=True, include_metrics=args.metrics)
    except FileNotFoundError:
        print(f"{RED}Error: File '{args.file}' not found.{RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{RED}Error reading file: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(format_banner(f"ITD-Scanner Report: {args.file}"))

    if report.get("syntax_error"):
        print(f"{YELLOW}[!] Warning: File contains syntax errors; static parse was incomplete.{RESET}\n")

    print(f"\n{BOLD}Idiomatic Technical Debt (ITD) Smells:{RESET}")
    total_smells = 0
    for smell, detected in report["flags"].items():
        if detected:
            total_smells += 1
            status = f"{RED}{BOLD}[DETECTED]{RESET}"
        else:
            status = f"{GREEN}[CLEAN]{RESET}"
        print(f"  {smell:<22}: {status}")

    if report["findings"]:
        print(f"\n{BOLD}Detailed Findings ({len(report['findings'])}):{RESET}")
        for f in report["findings"]:
            print(f"  {YELLOW}Line {f['line']:<3}{RESET} {BOLD}[{f['smell']}]{RESET}: {f['message']}")
    else:
        print(f"\n{GREEN}✓ No idiomatic technical debt detected. Code adheres to vectorized patterns.{RESET}")

    if args.metrics and "metrics" in report:
        m = report["metrics"]
        print(f"\n{BOLD}Software Complexity & Maintainability (Radon):{RESET}")
        print(f"  {GRAY}Lines of Code (LOC)      :{RESET} {m.get('loc')}")
        print(f"  {GRAY}Source Lines (SLOC)      :{RESET} {m.get('sloc')}")
        print(f"  {GRAY}Cyclomatic Complexity (CC):{RESET} {BOLD}{m.get('cyclomatic_complexity')}{RESET}")
        print(f"  {GRAY}Maintainability Index    :{RESET} {m.get('maintainability_index')}/100")
        if "halstead_volume" in m:
            print(f"  {GRAY}Halstead Volume          :{RESET} {m.get('halstead_volume')} bits")
            print(f"  {GRAY}Halstead Difficulty      :{RESET} {m.get('halstead_difficulty')}")

    status_footer = f"{RED}{BOLD}{total_smells} ITD Smells Found{RESET}" if total_smells > 0 else f"{GREEN}{BOLD}All Checks Passed{RESET}"
    print(f"\n{CYAN}{'='*65}{RESET}")
    print(f" Status: {status_footer}")
    print(f"{CYAN}{'='*65}{RESET}\n")

if __name__ == "__main__":
    main()
