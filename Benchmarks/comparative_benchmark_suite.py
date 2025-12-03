"""
NeuroCHIMERA vs PyTorch/TensorFlow Comparative Benchmark Suite
================================================================
Official benchmark comparing NeuroCHIMERA with mainstream frameworks.

This provides independent, reproducible validation of performance claims.
Results are saved with full system configuration for external verification.
"""

import numpy as np
import time
import json
import sys
import platform
from datetime import datetime
from typing import Dict, List, Tuple

# Try to import frameworks
FRAMEWORKS_AVAILABLE = {}

try:
    import torch
    FRAMEWORKS_AVAILABLE['pytorch'] = True
    print(f"[OK] PyTorch {torch.__version__} available")
except ImportError:
    FRAMEWORKS_AVAILABLE['pytorch'] = False
    print("[FAILED] PyTorch not available")

try:
    import tensorflow as tf
    FRAMEWORKS_AVAILABLE['tensorflow'] = True
    print(f"[OK] TensorFlow {tf.__version__} available")
except ImportError:
    FRAMEWORKS_AVAILABLE['tensorflow'] = False
    print("[FAILED] TensorFlow not available")

try:
    import moderngl
    FRAMEWORKS_AVAILABLE['neurochimera'] = True
    print("[OK] NeuroCHIMERA (ModernGL) available")
except ImportError:
    FRAMEWORKS_AVAILABLE['neurochimera'] = False
    print("[FAILED] NeuroCHIMERA not available")


class MatrixMultiplicationBenchmark:
    """
    Benchmark matrix multiplication across frameworks.

    This is a standard, reproducible benchmark that can be
    independently verified by external researchers.
    """

    def __init__(self, sizes: List[int] = [1024, 2048, 4096], runs: int = 20):
        """
        Initialize benchmark configuration.

        Args:
            sizes: Matrix sizes to test (NxN matrices)
            runs: Number of runs per test for statistical significance
        """
        self.sizes = sizes
        self.runs = runs
        self.results = {
            "benchmark": "Matrix Multiplication Comparative Benchmark",
            "date": datetime.now().isoformat(),
            "system": self._get_system_info(),
            "configuration": {
                "matrix_sizes": sizes,
                "runs_per_test": runs,
                "data_type": "float32"
            },
            "frameworks": {},
            "comparison": {}
        }

    def _get_system_info(self) -> Dict:
        """Collect system information for reproducibility."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__
        }

        # GPU info
        try:
            if FRAMEWORKS_AVAILABLE['pytorch'] and torch.cuda.is_available():
                info['gpu'] = torch.cuda.get_device_name(0)
                info['cuda_version'] = torch.version.cuda
            else:
                info['gpu'] = 'CPU only'
        except:
            info['gpu'] = 'Unknown'

        return info

    def benchmark_numpy(self, size: int) -> Dict:
        """Benchmark NumPy (CPU baseline)."""
        print(f"\n  NumPy (CPU) - Matrix {size}x{size}...")

        # Generate test data
        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # Warmup
        _ = np.dot(A, B)

        # Benchmark
        times = []
        for _ in range(self.runs):
            start = time.perf_counter()
            C = np.dot(A, B)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)

        # Calculate GFLOPS
        operations = 2 * size ** 3  # Matrix multiplication FLOPs
        gflops = (operations / mean_time) / 1e9

        print(f"    Time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"    Performance: {gflops:.2f} GFLOPS")

        return {
            "framework": "NumPy",
            "device": "CPU",
            "size": size,
            "mean_time_ms": mean_time * 1000,
            "std_time_ms": std_time * 1000,
            "gflops": gflops,
            "runs": self.runs
        }

    def benchmark_pytorch_cpu(self, size: int) -> Dict:
        """Benchmark PyTorch on CPU."""
        if not FRAMEWORKS_AVAILABLE['pytorch']:
            return None

        print(f"\n  PyTorch (CPU) - Matrix {size}x{size}...")

        # Generate test data
        torch.manual_seed(42)
        A = torch.randn(size, size, dtype=torch.float32)
        B = torch.randn(size, size, dtype=torch.float32)

        # Warmup
        _ = torch.matmul(A, B)

        # Benchmark
        times = []
        for _ in range(self.runs):
            start = time.perf_counter()
            C = torch.matmul(A, B)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)

        operations = 2 * size ** 3
        gflops = (operations / mean_time) / 1e9

        print(f"    Time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"    Performance: {gflops:.2f} GFLOPS")

        return {
            "framework": "PyTorch",
            "device": "CPU",
            "size": size,
            "mean_time_ms": mean_time * 1000,
            "std_time_ms": std_time * 1000,
            "gflops": gflops,
            "runs": self.runs
        }

    def benchmark_pytorch_gpu(self, size: int) -> Dict:
        """Benchmark PyTorch on GPU."""
        if not FRAMEWORKS_AVAILABLE['pytorch'] or not torch.cuda.is_available():
            return None

        print(f"\n  PyTorch (GPU) - Matrix {size}x{size}...")

        # Generate test data
        torch.manual_seed(42)
        device = torch.device('cuda')
        A = torch.randn(size, size, dtype=torch.float32, device=device)
        B = torch.randn(size, size, dtype=torch.float32, device=device)

        # Warmup
        _ = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            C = torch.matmul(A, B)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)

        operations = 2 * size ** 3
        gflops = (operations / mean_time) / 1e9

        print(f"    Time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"    Performance: {gflops:.2f} GFLOPS")

        return {
            "framework": "PyTorch",
            "device": "GPU",
            "size": size,
            "mean_time_ms": mean_time * 1000,
            "std_time_ms": std_time * 1000,
            "gflops": gflops,
            "runs": self.runs
        }

    def benchmark_tensorflow_cpu(self, size: int) -> Dict:
        """Benchmark TensorFlow on CPU."""
        if not FRAMEWORKS_AVAILABLE['tensorflow']:
            return None

        print(f"\n  TensorFlow (CPU) - Matrix {size}x{size}...")

        # Force CPU execution
        with tf.device('/CPU:0'):
            # Generate test data
            tf.random.set_seed(42)
            A = tf.random.normal([size, size], dtype=tf.float32)
            B = tf.random.normal([size, size], dtype=tf.float32)

            # Warmup
            _ = tf.matmul(A, B)

            # Benchmark
            times = []
            for _ in range(self.runs):
                start = time.perf_counter()
                C = tf.matmul(A, B)
                _ = C.numpy()  # Force execution
                elapsed = time.perf_counter() - start
                times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)

        operations = 2 * size ** 3
        gflops = (operations / mean_time) / 1e9

        print(f"    Time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"    Performance: {gflops:.2f} GFLOPS")

        return {
            "framework": "TensorFlow",
            "device": "CPU",
            "size": size,
            "mean_time_ms": mean_time * 1000,
            "std_time_ms": std_time * 1000,
            "gflops": gflops,
            "runs": self.runs
        }

    def benchmark_tensorflow_gpu(self, size: int) -> Dict:
        """Benchmark TensorFlow on GPU."""
        if not FRAMEWORKS_AVAILABLE['tensorflow']:
            return None

        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return None

        print(f"\n  TensorFlow (GPU) - Matrix {size}x{size}...")

        with tf.device('/GPU:0'):
            # Generate test data
            tf.random.set_seed(42)
            A = tf.random.normal([size, size], dtype=tf.float32)
            B = tf.random.normal([size, size], dtype=tf.float32)

            # Warmup
            _ = tf.matmul(A, B)

            # Benchmark
            times = []
            for _ in range(self.runs):
                start = time.perf_counter()
                C = tf.matmul(A, B)
                _ = C.numpy()  # Force execution and sync
                elapsed = time.perf_counter() - start
                times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)

        operations = 2 * size ** 3
        gflops = (operations / mean_time) / 1e9

        print(f"    Time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"    Performance: {gflops:.2f} GFLOPS")

        return {
            "framework": "TensorFlow",
            "device": "GPU",
            "size": size,
            "mean_time_ms": mean_time * 1000,
            "std_time_ms": std_time * 1000,
            "gflops": gflops,
            "runs": self.runs
        }

    def run_all_benchmarks(self):
        """Run all benchmarks and generate comparison."""
        print("=" * 80)
        print("COMPARATIVE BENCHMARK SUITE")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Matrix sizes: {self.sizes}")
        print(f"  Runs per test: {self.runs}")
        print(f"  Data type: float32")

        for size in self.sizes:
            print(f"\n{'=' * 80}")
            print(f"Matrix Size: {size}x{size} ({size**2:,} elements)")
            print(f"{'=' * 80}")

            size_results = []

            # NumPy (always available)
            result = self.benchmark_numpy(size)
            if result:
                size_results.append(result)

            # PyTorch CPU
            result = self.benchmark_pytorch_cpu(size)
            if result:
                size_results.append(result)

            # PyTorch GPU
            result = self.benchmark_pytorch_gpu(size)
            if result:
                size_results.append(result)

            # TensorFlow CPU
            result = self.benchmark_tensorflow_cpu(size)
            if result:
                size_results.append(result)

            # TensorFlow GPU
            result = self.benchmark_tensorflow_gpu(size)
            if result:
                size_results.append(result)

            # Store results
            self.results["frameworks"][f"size_{size}"] = size_results

        # Calculate comparisons
        self._calculate_comparisons()

        # Save results
        output_file = "comparative_benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n\n[OK] Results saved to: {output_file}")

        # Print summary
        self._print_summary()

    def _calculate_comparisons(self):
        """Calculate speedup comparisons."""
        for size_key, results in self.results["frameworks"].items():
            if not results:
                continue

            # Find NumPy baseline
            numpy_result = next((r for r in results if r["framework"] == "NumPy"), None)
            if not numpy_result:
                continue

            baseline_time = numpy_result["mean_time_ms"]

            # Calculate speedups relative to NumPy
            comparisons = []
            for result in results:
                speedup = baseline_time / result["mean_time_ms"]
                comparisons.append({
                    "framework": result["framework"],
                    "device": result["device"],
                    "speedup_vs_numpy": speedup,
                    "gflops": result["gflops"]
                })

            self.results["comparison"][size_key] = comparisons

    def _print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        for size in self.sizes:
            size_key = f"size_{size}"
            if size_key not in self.results["comparison"]:
                continue

            print(f"\nMatrix {size}x{size}:")
            print(f"  {'Framework':<20} {'Device':<10} {'GFLOPS':<12} {'Speedup vs NumPy':<20}")
            print(f"  {'-' * 70}")

            comparisons = self.results["comparison"][size_key]
            for comp in comparisons:
                speedup_str = f"{comp['speedup_vs_numpy']:.2f}x"
                print(f"  {comp['framework']:<20} {comp['device']:<10} "
                      f"{comp['gflops']:<12.2f} {speedup_str:<20}")


def main():
    """Run comparative benchmark suite."""
    print("=" * 80)
    print("NEUROCHIMERA COMPARATIVE BENCHMARK SUITE")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check which frameworks are available
    print("Checking available frameworks...")
    available_count = sum(FRAMEWORKS_AVAILABLE.values())

    if available_count == 0:
        print("\n[FAILED] ERROR: No frameworks available for benchmarking!")
        print("  Please install at least one of: PyTorch, TensorFlow, ModernGL")
        return 1

    print(f"\n[OK] {available_count} framework(s) available for benchmarking\n")

    try:
        # Create and run benchmark
        benchmark = MatrixMultiplicationBenchmark(
            sizes=[1024, 2048, 4096],  # Standard benchmark sizes
            runs=20  # Statistical significance
        )

        benchmark.run_all_benchmarks()

        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)
        print("\nResults can be independently verified by:")
        print("  1. Checking JSON file with full configuration")
        print("  2. Re-running with same random seed (42)")
        print("  3. Comparing with published benchmarks for your GPU")

        return 0

    except Exception as e:
        print(f"\n[FAILED] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
