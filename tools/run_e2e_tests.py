#!/usr/bin/env python3
"""
ASL Dynamic Gesture Recognition E2E Test Suite Runner
======================================================
Automated test runner executing the complete 4-tier test pyramid and verification plan:
  - Verification Plan (Steps 1-5 from ORIGINAL_REQUEST.md)
  - Tier 1: Feature Completeness & Primary Behaviors (>= 5 tests per feature)
  - Tier 2: Boundary Conditions, Physical Outliers & Negative Rejections
  - Tier 3: Cross-Feature Pairwise Interactions & Concurrency Lock Guards
  - Tier 4: Realistic End-to-End User Journeys, Workloads & Stress Tests

Usage:
  python run_e2e_tests.py
  python run_e2e_tests.py --tier 1
  python run_e2e_tests.py --tier 2
  python run_e2e_tests.py --tier 3
  python run_e2e_tests.py --tier 4
  python run_e2e_tests.py --steps
  python run_e2e_tests.py -v
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import os
import sys
import time
import argparse
import subprocess
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


TIER_DEFINITIONS = [
    {
        "id": "steps",
        "name": "Verification Steps 1-5",
        "file": "tests/test_verification_steps.py",
        "desc": "System architecture, FSM spotting, DTW matching, TTS latency, and HUD rendering"
    },
    {
        "id": "tier1",
        "name": "Tier 1: Feature Coverage",
        "file": "tests/test_tier1_features.py",
        "desc": "5 vocabulary words (>=5 tests each), FSM spotting, Sakoe-Chiba DTW, HUD, Swipe-Left"
    },
    {
        "id": "tier2",
        "name": "Tier 2: Boundary & Corner Cases",
        "file": "tests/test_tier2_boundaries.py",
        "desc": "Stationary hand, velocity thresholds, duration gates, palm floors, occlusion grace"
    },
    {
        "id": "tier3",
        "name": "Tier 3: Pairwise Interactions",
        "file": "tests/test_tier3_interactions.py",
        "desc": "Swipe interruption, silence commit, concurrent TTS, debounce guard, backspace"
    },
    {
        "id": "tier4",
        "name": "Tier 4: Realistic Scenarios",
        "file": "tests/test_tier4_scenarios.py",
        "desc": "Multi-word sentence generation, edit-and-correct, noise rejection, 30s stress test"
    }
]


def run_test_file(test_file: str, verbose: bool = False) -> Dict[str, Any]:
    """Runs a single test file using pytest and parses pass/fail counts and timing."""
    cmd = [sys.executable, "-m", "pytest", test_file, "-q", "--tb=short"]
    if verbose:
        cmd.append("-v")

    t_start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    duration = time.perf_counter() - t_start

    stdout = proc.stdout
    passed = 0
    failed = 0
    errors = 0

    # Parse pytest summary line (e.g., '10 passed in 0.45s', '1 failed, 4 passed in 0.60s')
    lines = stdout.strip().splitlines()
    summary_line = lines[-1] if lines else ""

    for token in summary_line.split(","):
        token = token.strip()
        if "passed" in token:
            try:
                passed = int(token.split()[0])
            except ValueError:
                pass
        elif "failed" in token:
            try:
                failed = int(token.split()[0])
            except ValueError:
                pass
        elif "error" in token:
            try:
                errors = int(token.split()[0])
            except ValueError:
                pass

    return {
        "file": test_file,
        "exit_code": proc.returncode,
        "passed": passed,
        "failed": failed + errors,
        "total": passed + failed + errors,
        "duration": duration,
        "stdout": stdout,
        "stderr": proc.stderr
    }


def print_banner():
    banner = r"""
================================================================================
   ASL REAL-TIME DYNAMIC GESTURE RECOGNITION & SENTENCE BUILDER PIPELINE
                     END-TO-END AUTOMATED TEST RUNNER
================================================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="ASL Dynamic Gesture E2E Test Suite Runner")
    parser.add_argument("--tier", choices=["1", "2", "3", "4"], help="Run only the specified tier (1, 2, 3, or 4)")
    parser.add_argument("--steps", action="store_true", help="Run only the Verification Plan steps 1-5")
    parser.add_argument("-v", "--verbose", action="store_true", help="Display verbose pytest output")
    args = parser.parse_args()

    print_banner()

    # Filter target tiers based on arguments
    targets = []
    if args.steps:
        targets = [t for t in TIER_DEFINITIONS if t["id"] == "steps"]
    elif args.tier:
        tier_key = f"tier{args.tier}"
        targets = [t for t in TIER_DEFINITIONS if t["id"] == tier_key]
    else:
        targets = list(TIER_DEFINITIONS)

    results = []
    total_passed = 0
    total_failed = 0
    total_duration = 0.0

    print(f"Executing {len(targets)} test target suite(s)...")
    print("-" * 80)

    for target in targets:
        print(f"[*] Running {target['name']} ({target['file']})...", flush=True)
        res = run_test_file(target["file"], verbose=args.verbose)
        results.append((target, res))

        total_passed += res["passed"]
        total_failed += res["failed"]
        total_duration += res["duration"]

        status_str = "[PASS]" if res["exit_code"] == 0 else "[FAIL]"
        print(f"    -> {status_str} {res['passed']}/{res['total']} passed in {res['duration']:.2f}s")
        if res["exit_code"] != 0 and args.verbose:
            print("\n--- Failure Details ---")
            print(res["stdout"])
            print(res["stderr"])
            print("-" * 80)

    # Formatted Summary Table
    print("\n" + "=" * 80)
    print(f"{'SUITE / TIER':<32} {'PASSED':<10} {'FAILED':<10} {'TIME (s)':<12} {'STATUS':<10}")
    print("-" * 80)

    for target, res in results:
        status_text = "PASSED" if res["exit_code"] == 0 else "FAILED"
        print(f"{target['name']:<32} {res['passed']:<10} {res['failed']:<10} {res['duration']:<12.2f} {status_text:<10}")

    print("=" * 80)
    overall_status = "ALL TESTS PASSED" if total_failed == 0 else f"{total_failed} TESTS FAILED"
    print(f"TOTAL: {total_passed} Passed, {total_failed} Failed across {len(results)} suites in {total_duration:.2f}s")
    print(f"VERDICT: {overall_status}")
    print("=" * 80 + "\n")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
