"""
Master Benchmark Execution Script
==================================
Runs all benchmarks sequentially and generates visualizations.

This script handles:
- GPU HNS benchmarks with statistical significance
- Comparative benchmarks with PyTorch/TensorFlow
- Automatic visualization generation
- Error handling and logging
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")

def run_command(cmd, description):
    """
    Run a command and return success status.

    Args:
        cmd: Command to run
        description: Description for logging

    Returns:
        True if successful, False otherwise
    """
    print(f"\n[*] {description}...")
    print(f"    Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"    [SUCCESS] Completed in {elapsed:.2f}s")
            return True
        else:
            print(f"    [FAILED] Exit code: {result.returncode}")
            if result.stderr:
                print(f"    Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] Command exceeded 10 minute limit")
        return False
    except Exception as e:
        print(f"    [ERROR] {str(e)}")
        return False

def main():
    """Run all benchmarks."""
    print_header("NEUROCHIMERA COMPLETE BENCHMARK SUITE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {Path.cwd()}")

    results = {}

    # 1. GPU HNS Benchmarks
    print_header("STEP 1: GPU HNS Benchmarks")
    success = run_command(
        [sys.executable, "gpu_hns_complete_benchmark.py"],
        "Running GPU HNS complete benchmark suite"
    )
    results["gpu_hns"] = success

    if success:
        print("    Output: gpu_hns_complete_benchmark_results.json")

    # 2. Comparative Benchmarks
    print_header("STEP 2: Comparative Framework Benchmarks")
    success = run_command(
        [sys.executable, "comparative_benchmark_suite.py"],
        "Running PyTorch/TensorFlow comparative benchmarks"
    )
    results["comparative"] = success

    if success:
        print("    Output: comparative_benchmark_results.json")

    # 3. Visualizations
    print_header("STEP 3: Generating Visualizations")
    success = run_command(
        [sys.executable, "visualize_benchmarks.py"],
        "Generating benchmark visualizations"
    )
    results["visualizations"] = success

    if success:
        print("    Output: benchmark_graphs/*.png")

    # Summary
    print_header("BENCHMARK EXECUTION SUMMARY")

    total = len(results)
    successful = sum(results.values())
    failed = total - successful

    print(f"Total benchmarks: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    for name, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {status} {name}")

    print()

    if failed == 0:
        print("[SUCCESS] All benchmarks completed successfully!")
        print("\nResults available in:")
        print("  - GPU HNS: gpu_hns_complete_benchmark_results.json")
        print("  - Comparative: comparative_benchmark_results.json")
        print("  - Visualizations: benchmark_graphs/")
        return 0
    else:
        print(f"[WARNING] {failed} benchmark(s) failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"\nExecution completed with code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Benchmark execution cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[CRITICAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
