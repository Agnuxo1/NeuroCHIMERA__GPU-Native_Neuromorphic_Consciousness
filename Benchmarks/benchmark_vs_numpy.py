"""
CHIMERA vs NumPy Benchmark
===========================
Baseline comparison against NumPy (CPU).

This establishes the CPU baseline for comparison.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import NeuroCHIMERAConfig
from engine_multi_core import MultiCoreNeuroCHIMERA


def benchmark_numpy_operations(size: int, iterations: int = 20) -> dict:
    """Benchmark NumPy matrix operations."""
    print(f"\nNumPy - {size}×{size} matrix operations")
    
    # Create random matrices
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    # Warmup
    _ = A @ B
    
    # Benchmark matrix multiplication
    times_matmul = []
    for i in range(iterations):
        start = time.perf_counter()
        C = A @ B
        elapsed = time.perf_counter() - start
        times_matmul.append(elapsed)
    
    # Benchmark element-wise operations
    times_elemwise = []
    for i in range(iterations):
        start = time.perf_counter()
        C = A * B + A
        C = np.tanh(C)
        elapsed = time.perf_counter() - start
        times_elemwise.append(elapsed)
    
    # Benchmark convolution-like operation (simulates neural neighborhood)
    kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    from scipy.ndimage import convolve
    times_conv = []
    for i in range(iterations):
        start = time.perf_counter()
        C = convolve(A, kernel, mode='constant')
        elapsed = time.perf_counter() - start
        times_conv.append(elapsed)
    
    return {
        'framework': 'NumPy',
        'device': 'CPU',
        'size': size,
        'operations': {
            'matrix_multiply': {
                'mean_time_ms': np.mean(times_matmul) * 1000,
                'std_time_ms': np.std(times_matmul) * 1000,
                'gflops': (2 * size**3) / (np.mean(times_matmul) * 1e9)
            },
            'elementwise': {
                'mean_time_ms': np.mean(times_elemwise) * 1000,
                'std_time_ms': np.std(times_elemwise) * 1000,
            },
            'convolution': {
                'mean_time_ms': np.mean(times_conv) * 1000,
                'std_time_ms': np.std(times_conv) * 1000,
            }
        }
    }


def benchmark_chimera_operations(neurons: int, iterations: int = 20) -> dict:
    """Benchmark CHIMERA equivalent operations."""
    print(f"\nCHIMERA - {neurons:,} neurons")
    
    try:
        config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
        engine = MultiCoreNeuroCHIMERA(config=config, parallel_batches=1)
        
        if engine.ctx is None:
            return {'error': 'GPU not available'}
        
        # Warmup
        engine.evolve_ultra_optimized(iterations=3)
        engine.ctx.finish()
        
        # Benchmark evolution (includes matrix-like operations)
        times_evolve = []
        for i in range(iterations):
            start = time.perf_counter()
            engine.evolve_ultra_optimized(iterations=1)
            engine.ctx.finish()
            elapsed = time.perf_counter() - start
            times_evolve.append(elapsed)
        
        mean_time = np.mean(times_evolve)
        throughput = config.neurons / mean_time
        gflops = (config.neurons * 25) / (mean_time * 1e9)
        
        engine.release()
        
        return {
            'framework': 'CHIMERA',
            'device': 'GPU (OpenGL)',
            'neurons': config.neurons,
            'texture_size': config.texture_size,
            'operations': {
                'evolution': {
                    'mean_time_ms': mean_time * 1000,
                    'std_time_ms': np.std(times_evolve) * 1000,
                    'throughput_neurons_per_sec': throughput,
                    'gflops': gflops
                }
            }
        }
    
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    """Run CHIMERA vs NumPy comparison."""
    print("\n" + "="*80)
    print("CHIMERA vs NumPy Baseline Comparison")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print("="*80)
    
    results = {
        'benchmark': 'CHIMERA vs NumPy',
        'timestamp': datetime.now().isoformat(),
        'numpy_results': [],
        'chimera_results': []
    }
    
    # Test configurations
    test_sizes = [
        (1024, 1_048_576),   # 1024×1024 matrix ≈ 1M neurons
        (2048, 4_194_304),   # 2048×2048 matrix ≈ 4M neurons
    ]
    
    for matrix_size, neurons in test_sizes:
        print(f"\n{'='*80}")
        print(f"Testing: {matrix_size}×{matrix_size} / {neurons:,} neurons")
        print(f"{'='*80}")
        
        # NumPy baseline
        numpy_result = benchmark_numpy_operations(matrix_size, iterations=20)
        results['numpy_results'].append(numpy_result)
        
        print(f"\nNumPy Results:")
        print(f"  Matrix Multiply: {numpy_result['operations']['matrix_multiply']['mean_time_ms']:.2f}ms "
              f"({numpy_result['operations']['matrix_multiply']['gflops']:.2f} GFLOPS)")
        print(f"  Element-wise: {numpy_result['operations']['elementwise']['mean_time_ms']:.2f}ms")
        print(f"  Convolution: {numpy_result['operations']['convolution']['mean_time_ms']:.2f}ms")
        
        # CHIMERA
        chimera_result = benchmark_chimera_operations(neurons, iterations=20)
        results['chimera_results'].append(chimera_result)
        
        if 'error' not in chimera_result:
            print(f"\nCHIMERA Results:")
            print(f"  Evolution: {chimera_result['operations']['evolution']['mean_time_ms']:.2f}ms "
                  f"({chimera_result['operations']['evolution']['gflops']:.3f} GFLOPS)")
            print(f"  Throughput: {chimera_result['operations']['evolution']['throughput_neurons_per_sec']/1e6:.2f}M neurons/s")
    
    # Save results
    output_file = Path(__file__).parent / 'benchmark_vs_numpy_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"[OK] Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Comparison summary
    print("\n" + "="*80)
    print("SUMMARY - CHIMERA vs NumPy")
    print("="*80)
    print("\nNote: Direct comparison is approximate as CHIMERA performs")
    print("neuromorphic cellular automata evolution, while NumPy performs")
    print("standard matrix operations. Both are compute-intensive GPU/CPU tasks.")
    print("\nCHIMERA runs on GPU (OpenGL), NumPy runs on CPU.")
    print("="*80)


if __name__ == '__main__':
    main()
