#!/usr/bin/env python3
"""
Auto-tune all candidate solutions

Runs autotuning for all candidates and compares the best configurations.
"""

import subprocess
import sys
import re
from pathlib import Path


def run_autotune(script_name):
    """Run an autotune script and capture output, extracting best result"""
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )

        # Print the output
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Parse output to extract optimal BLOCK_SIZE and time
        optimal_block_size = None
        optimal_time = None

        # Look for "Optimal BLOCK_SIZE: XXX (Y.YYYYms)"
        match = re.search(r'Optimal BLOCK_SIZE: (\d+) \(([0-9.]+)ms\)', result.stdout)
        if match:
            optimal_block_size = int(match.group(1))
            optimal_time = float(match.group(2))

        return True, optimal_block_size, optimal_time
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False, None, None


def main():
    print("=" * 80)
    print("AUTO-TUNING ALL CANDIDATES")
    print("=" * 80)
    print("\nThis will run autotuning for all candidate solutions.")
    print("Each candidate will be tested with various BLOCK_SIZE values.\n")

    # Find all autotune scripts in candidate folders
    candidates_dir = Path(__file__).parent / "candidates"
    candidate_dirs = sorted([d for d in candidates_dir.iterdir() if d.is_dir()])

    autotune_scripts = []
    for candidate_dir in candidate_dirs:
        autotune_file = candidate_dir / "autotune.py"
        if autotune_file.exists():
            autotune_scripts.append(autotune_file)

    if not autotune_scripts:
        print("No autotune scripts found!")
        return 1

    print(f"Found {len(autotune_scripts)} candidate solutions with autotune scripts:")
    for script in autotune_scripts:
        candidate_name = script.parent.name
        print(f"  - {candidate_name}")
    print()

    # Run each autotune script
    results = {}
    for script in autotune_scripts:
        candidate_name = script.parent.name
        success, block_size, time_ms = run_autotune(script)
        results[candidate_name] = {
            'success': success,
            'block_size': block_size,
            'time_ms': time_ms
        }

    # Summary
    print("\n" + "=" * 80)
    print("AUTOTUNE SUMMARY - CANDIDATE COMPARISON")
    print("=" * 80)
    print()

    # Collect successful results
    successful_results = [(name, data) for name, data in results.items() if data['success'] and data['time_ms'] is not None]

    if successful_results:
        # Find best time for speedup calculation
        best_time = min(data['time_ms'] for _, data in successful_results)

        print(f"{'Candidate':<30} {'Status':<10} {'Best BLOCK_SIZE':<18} {'Avg Time (ms)':<15} {'Speedup':<10}")
        print("-" * 95)

        for candidate_name in sorted(results.keys()):
            data = results[candidate_name]
            if data['success'] and data['time_ms'] is not None:
                speedup = best_time / data['time_ms']
                marker = " <- FASTEST" if data['time_ms'] == best_time else ""
                print(f"{candidate_name:<30} {'SUCCESS':<10} {data['block_size']:<18} {data['time_ms']:<15.4f} {speedup:<10.2f}x{marker}")
            elif data['success']:
                print(f"{candidate_name:<30} {'SUCCESS':<10} {'N/A':<18} {'N/A':<15} {'N/A':<10}")
            else:
                print(f"{candidate_name:<30} {'FAILED':<10} {'N/A':<18} {'N/A':<15} {'N/A':<10}")

        print()
        print("=" * 80)
        best_candidate = min(successful_results, key=lambda x: x[1]['time_ms'])
        print(f"FASTEST: {best_candidate[0]} with BLOCK_SIZE={best_candidate[1]['block_size']} ({best_candidate[1]['time_ms']:.4f}ms)")
        print("=" * 80)
    else:
        print("No successful autotune runs to compare!")
        print()
        for candidate_name, data in results.items():
            status = "SUCCESS" if data['success'] else "FAILED"
            print(f"  {candidate_name:<30} {status}")

    print()
    print("=" * 80)
    print("Next steps:")
    print("  1. Update the BLOCK_SIZE in each candidate's solution.py with values above")
    print("  2. Run 'python validate.py' to validate all candidates")
    print("  3. Update submission.py to import the fastest candidate")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
