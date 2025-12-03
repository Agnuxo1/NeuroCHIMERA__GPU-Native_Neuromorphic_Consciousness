"""
GPU Utilization Benchmark - Optimized Implementation
====================================================

Tests optimized implementation to verify GPU utilization improvements.
"""

import time
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine_optimized import OptimizedNeuroCHIMERA
from engine import NeuroCHIMERAConfig


def benchmark_size(neurons: int, iterations: int = 10):
    """Benchmark a specific network size."""
    print(f"\n{'='*80}")
    print(f"Benchmarking {neurons:,} neurons")
    print(f"{'='*80}")
    
    try:
        config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
        brain = OptimizedNeuroCHIMERA(config=config)
        
        if brain.ctx is None:
            print("  [FAIL] GPU context not available")
            return None
        
        size = config.texture_size
        print(f"  Texture: {size}×{size}")
        print(f"  Actual neurons: {config.neurons:,}")
        
        # Calculate memory
        bytes_per_pixel = 16
        total_mem = (size * size * bytes_per_pixel * 4) / 1024 / 1024  # 4 textures
        print(f"  Memory: {total_mem:.1f} MB")
        
        # Warmup
        print("  Warming up...")
        brain.evolve_optimized(iterations=1)
        brain.ctx.finish()
        
        # Benchmark
        print(f"  Running {iterations} iterations...")
        times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            result = brain.evolve_optimized(iterations=1)
            brain.ctx.finish()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            if (i + 1) % 5 == 0:
                print(f"    Iteration {i+1}/{iterations}: {elapsed*1000:.2f}ms")
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        neurons_per_sec = config.neurons / avg_time
        gflops = (config.neurons * 25) / (avg_time * 1e9)
        
        print(f"\n  [OK] Results:")
        print(f"    Avg time: {avg_time*1000:.2f}ms")
        print(f"    Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
        print(f"    Std dev: {std_time*1000:.2f}ms")
        print(f"    Throughput: {neurons_per_sec/1e6:.2f}M neurons/s")
        print(f"    Compute: {gflops:.2f} GFLOPS")
        
        brain.release()
        
        return {
            'neurons': config.neurons,
            'texture_size': size,
            'memory_mb': total_mem,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': std_time,
            'neurons_per_second': neurons_per_sec,
            'gflops': gflops
        }
        
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run GPU utilization benchmark."""
    print("\n" + "="*80)
    print("GPU UTILIZATION BENCHMARK - Optimized Implementation")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTesting optimized implementation with:")
    print("  - 32×32 work groups (1024 threads per group)")
    print("  - Pipelined iterations")
    print("  - Pre-bound resources")
    print("  - Optimized memory access")
    print("="*80)
    
    # Test sizes
    test_sizes = [
        1_048_576,      # 1M
        4_194_304,      # 4M
        16_777_216,     # 16M
        67_108_864,     # 67M
    ]
    
    results = []
    
    for neurons in test_sizes:
        result = benchmark_size(neurons, iterations=10)
        if result:
            results.append(result)
        else:
            print(f"\n  Stopping at {neurons:,} neurons due to failure")
            break
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if results:
        print("\n| Neurons | Texture | Memory (MB) | Time (ms) | Throughput (M/s) | GFLOPS |")
        print("|---------|---------|-------------|-----------|------------------|--------|")
        
        for r in results:
            print(f"| {r['neurons']:,} | {r['texture_size']}×{r['texture_size']} | "
                  f"{r['memory_mb']:.1f} | {r['avg_time']*1000:.2f} | "
                  f"{r['neurons_per_second']/1e6:.2f} | {r['gflops']:.2f} |")
        
        # Find best performance
        best_throughput = max(results, key=lambda x: x['neurons_per_second'])
        print(f"\n  Best throughput: {best_throughput['neurons_per_second']/1e6:.2f}M neurons/s")
        print(f"    at {best_throughput['neurons']:,} neurons ({best_throughput['texture_size']}×{best_throughput['texture_size']})")
        
        # Consistency check
        std_devs = [r['std_time'] / r['avg_time'] for r in results]
        avg_consistency = np.mean(std_devs) * 100
        print(f"\n  Consistency: {avg_consistency:.1f}% std dev (lower is better)")
        
        if avg_consistency < 10:
            print("    [OK] Very consistent performance")
        elif avg_consistency < 20:
            print("    [OK] Good consistency")
        else:
            print("    [WARN] Some variability in performance")
    
    print("\n" + "="*80)
    print("[OK] Benchmark completed")
    print("="*80)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("1. Monitor GPU usage with 'nvidia-smi' during execution")
    print("2. Check for consistent 70-80% GPU utilization (not 10% with spikes)")
    print("3. Verify smooth load without 100% spikes causing errors")
    print("4. Compare with previous benchmarks to measure improvement")


if __name__ == '__main__':
    main()

