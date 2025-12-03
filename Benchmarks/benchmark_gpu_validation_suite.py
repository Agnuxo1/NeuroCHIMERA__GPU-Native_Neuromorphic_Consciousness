"""
Complete GPU Validation Benchmark Suite
========================================
Comprehensive benchmark suite with GPU utilization validation.

Tests:
- Multiple scales (1M, 4M, 16M, 67M neurons)
- GPU utilization monitoring
- Performance metrics
- Scientific statistical analysis
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import NeuroCHIMERAConfig
from engine_optimized import OptimizedNeuroCHIMERA
from engine_multi_core import MultiCoreNeuroCHIMERA
from Benchmarks.gpu_utilization_validator import validate_benchmark, GPUUtilizationValidator


def benchmark_scale(neurons: int, engine_class, iterations: int = 20, 
                    **engine_kwargs) -> dict:
    """Benchmark a specific network scale."""
    print(f"\n{'='*80}")
    print(f"Benchmarking {neurons:,} neurons")
    print(f"{'='*80}")
    
    try:
        config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
        engine = engine_class(config=config, **engine_kwargs)
        
        if engine.ctx is None:
            print("  [SKIP] GPU not available")
            return None
        
        size = config.texture_size
        print(f"  Texture: {size}×{size}")
        print(f"  Actual neurons: {config.neurons:,}")
        
        # Memory calculation
        bytes_per_pixel = 16  # RGBA32F = 4 floats × 4 bytes
        num_textures = 4
        total_mem_mb = (size * size * bytes_per_pixel * num_textures) / (1024 * 1024)
        print(f"  GPU Memory: {total_mem_mb:.1f} MB")
        
        # Warmup
        print(f"  Warming up...")
        if hasattr(engine, 'evolve_ultra_optimized'):
            engine.evolve_ultra_optimized(iterations=3)
        else:
            engine.evolve_optimized(iterations=3)
        engine.ctx.finish()
        
        # Benchmark with timing
        print(f"  Running {iterations} iterations...")
        times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            
            if hasattr(engine, 'evolve_ultra_optimized'):
                result = engine.evolve_ultra_optimized(iterations=1)
            else:
                result = engine.evolve_optimized(iterations=1)
            
            engine.ctx.finish()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{iterations}: {elapsed*1000:.2f}ms")
        
        # Statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        throughput = config.neurons / mean_time
        gflops = (config.neurons * 25) / (mean_time * 1e9)  # ~25 FLOPs per neuron
        
        print(f"\n  Results:")
        print(f"    Mean: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"    Range: [{min_time*1000:.2f}, {max_time*1000:.2f}]ms")
        print(f"    Throughput: {throughput/1e6:.2f}M neurons/s")
        print(f"    Compute: {gflops:.3f} GFLOPS")
        
        engine.release()
        
        return {
            'neurons': config.neurons,
            'texture_size': size,
            'memory_mb': total_mem_mb,
            'iterations': iterations,
            'times_ms': [t * 1000 for t in times],
            'mean_time_ms': mean_time * 1000,
            'std_time_ms': std_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'throughput_neurons_per_sec': throughput,
            'gflops': gflops,
            'engine': engine_class.__name__
        }
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_with_gpu_monitoring(neurons: int, engine_class, iterations: int = 20,
                                   **engine_kwargs) -> dict:
    """Benchmark with GPU utilization monitoring."""
    
    def run_benchmark():
        return benchmark_scale(neurons, engine_class, iterations, **engine_kwargs)
    
    result = validate_benchmark(
        run_benchmark,
        f"{engine_class.__name__} - {neurons:,} neurons",
        target_min_util=60.0  # Lower target for smaller scales
    )
    
    return result


def main():
    """Run complete GPU validation benchmark suite."""
    print("\n" + "="*80)
    print("CHIMERA COMPLETE GPU VALIDATION BENCHMARK SUITE")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print("="*80)
    
    # Test configurations
    test_configs = [
        {'neurons': 1_048_576, 'iterations': 20},      # 1M
        {'neurons': 4_194_304, 'iterations': 15},      # 4M
        {'neurons': 16_777_216, 'iterations': 10},     # 16M
        {'neurons': 67_108_864, 'iterations': 5},      # 67M
    ]
    
    all_results = {
        'suite_name': 'Complete GPU Validation',
        'timestamp': datetime.now().isoformat(),
        'optimized_engine': [],
        'multi_core_engine': [],
        'multi_core_batch': []
    }
    
    # Test 1: Optimized Engine
    print("\n" + "="*80)
    print("TEST 1: Optimized Engine (32×32 work groups, pipelined)")
    print("="*80)
    
    for config in test_configs:
        result = benchmark_with_gpu_monitoring(
            config['neurons'],
            OptimizedNeuroCHIMERA,
            config['iterations']
        )
        if result:
            all_results['optimized_engine'].append(result)
    
    # Test 2: Multi-Core Engine (ultra-optimized)
    print("\n" + "="*80)
    print("TEST 2: Multi-Core Engine (ultra-optimized dispatch)")
    print("="*80)
    
    for config in test_configs:
        result = benchmark_with_gpu_monitoring(
            config['neurons'],
            MultiCoreNeuroCHIMERA,
            config['iterations'],
            parallel_batches=1  # Single for fair comparison
        )
        if result:
            all_results['multi_core_engine'].append(result)
    
    # Test 3: Multi-Core Batch Parallel
    print("\n" + "="*80)
    print("TEST 3: Multi-Core Batch Parallel (4× parallelism)")
    print("="*80)
    
    for config in test_configs[:2]:  # Only smaller scales for batch
        result = benchmark_with_gpu_monitoring(
            config['neurons'],
            MultiCoreNeuroCHIMERA,
            config['iterations'],
            parallel_batches=4
        )
        if result:
            all_results['multi_core_batch'].append(result)
    
    # Save results
    output_file = Path(__file__).parent / 'gpu_validation_suite_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"[OK] Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nOptimized Engine Performance:")
    for r in all_results['optimized_engine']:
        if 'benchmark_result' in r and r['benchmark_result']:
            br = r['benchmark_result']
            print(f"  {br['neurons']:>10,} neurons: {br['throughput_neurons_per_sec']/1e6:>6.2f}M/s  "
                  f"{br['gflops']:>6.3f} GFLOPS  "
                  f"({br['mean_time_ms']:>6.2f}ms)")
    
    print("\nMulti-Core Engine Performance:")
    for r in all_results['multi_core_engine']:
        if 'benchmark_result' in r and r['benchmark_result']:
            br = r['benchmark_result']
            print(f"  {br['neurons']:>10,} neurons: {br['throughput_neurons_per_sec']/1e6:>6.2f}M/s  "
                  f"{br['gflops']:>6.3f} GFLOPS  "
                  f"({br['mean_time_ms']:>6.2f}ms)")
    
    print("\nGPU Utilization Summary:")
    for test_name, results in [('Optimized', all_results['optimized_engine']),
                                ('Multi-Core', all_results['multi_core_engine'])]:
        if results:
            gpu_utils = []
            for r in results:
                if 'gpu_validation' in r and 'gpu_utilization' in r['gpu_validation']:
                    gpu_utils.append(r['gpu_validation']['gpu_utilization']['mean'])
            
            if gpu_utils:
                avg_util = np.mean(gpu_utils)
                print(f"  {test_name}: {avg_util:.1f}% average GPU utilization")
                
                if avg_util >= 80:
                    verdict = "EXCELLENT - Target achieved!"
                elif avg_util >= 60:
                    verdict = "GOOD - Above baseline"
                elif avg_util >= 30:
                    verdict = "MODERATE - Needs optimization"
                else:
                    verdict = "POOR - CPU fallback suspected"
                
                print(f"    Verdict: {verdict}")
    
    print("\n" + "="*80)
    print("[OK] GPU Validation Suite Complete")
    print("="*80)


if __name__ == '__main__':
    main()
