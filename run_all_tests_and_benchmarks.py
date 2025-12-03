"""
Master Script: Run All Tests and Benchmarks
===========================================

Execute complete testing and benchmarking workflow.
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and report results."""
    print("\n" + "=" * 80)
    print(f"{description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        print(f"\n[OK] Completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] Failed after {elapsed:.1f}s")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"[ERROR] Command not found: {cmd[0]}")
        return False


def main():
    """Run complete testing and benchmarking workflow."""
    print("\n" + "=" * 80)
    print("NEUROCHIMERA: COMPLETE TESTING AND BENCHMARKING WORKFLOW")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    base_dir = Path(__file__).parent
    results = {}
    
    # Phase 1: Core Component Tests
    print("\n" + "=" * 80)
    print("PHASE 1: CORE COMPONENT TESTS")
    print("=" * 80)
    
    test_files = [
        ("tests/test_core_components.py", "Core Component Tests"),
        ("tests/test_integration.py", "Integration Tests"),
        ("tests/test_consciousness_parameters.py", "Consciousness Parameter Tests"),
    ]
    
    for test_file, description in test_files:
        if (base_dir / test_file).exists():
            results[test_file] = run_command(
                [sys.executable, str(base_dir / test_file)],
                description
            )
        else:
            print(f"[SKIP] {test_file} not found")
            results[test_file] = None
    
    # Phase 2: Benchmarks
    print("\n" + "=" * 80)
    print("PHASE 2: PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    benchmark_files = [
        ("Benchmarks/gpu_hns_complete_benchmark.py", "GPU HNS Performance (Core)"),
        ("Benchmarks/gpu_saturation_benchmark.py", "GPU Saturation Test (Stress)"),
        ("Benchmarks/gpu_hns_precision_benchmark.py", "GPU HNS Precision Certification"),
        ("benchmarks/benchmark_neurochimera_system.py", "System Performance (Certified GPU)"),
    ]
    
    for bench_file, description in benchmark_files:
        if (base_dir / bench_file).exists():
            results[bench_file] = run_command(
                [sys.executable, str(base_dir / bench_file)],
                description
            )
        else:
            print(f"[SKIP] {bench_file} not found")
            results[bench_file] = None
    
    # Optional: Comparative benchmarks (requires PyTorch)
    comp_file = base_dir / "benchmarks/benchmark_comparative.py"
    if comp_file.exists():
        print("\n[INFO] Comparative benchmarks require PyTorch")
        print("       Run manually: python benchmarks/benchmark_comparative.py")
        results["benchmarks/benchmark_comparative.py"] = None
    
    # Phase 3: Generate Reports
    print("\n" + "=" * 80)
    print("PHASE 3: REPORT GENERATION")
    print("=" * 80)
    
    report_scripts = [
        ("generate_benchmark_report.py", "Benchmark Reports"),
        ("generate_consciousness_report.py", "Consciousness Reports"),
        ("create_visualizations.py", "Visualizations"),
        ("document_execution_results.py", "Comprehensive Documentation"),
    ]
    
    for script, description in report_scripts:
        script_path = base_dir / script
        if script_path.exists():
            results[script] = run_command(
                [sys.executable, str(script_path)],
                description
            )
        else:
            print(f"[SKIP] {script} not found")
            results[script] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print()
    
    if failed > 0:
        print("Failed components:")
        for name, result in results.items():
            if result is False:
                print(f"  - {name}")
        print()
    
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Review test outputs above")
    print("2. Check generated reports in reports/ directory")
    print("3. View visualizations in visualizations/ directory")
    print("4. Run simulations manually:")
    print("   python run_consciousness_emergence.py --neurons 65536 --epochs 1000")
    print("   python benchmark_complete_system.py --neurons 1048576 --epochs 100")
    print()
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

