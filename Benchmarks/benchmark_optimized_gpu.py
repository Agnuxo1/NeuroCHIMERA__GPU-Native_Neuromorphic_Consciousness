"""
Optimized GPU Benchmark
========================

Benchmark the optimized NeuroCHIMERA engine with high GPU utilization.
Compares optimized vs standard implementation.
"""

import time
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import NeuroCHIMERA, NeuroCHIMERAConfig
from engine_optimized import OptimizedNeuroCHIMERA


class OptimizedGPUBenchmark:
    """Benchmark optimized GPU implementation."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_comparison(self, neurons: int = 1048576, iterations: int = 50):
        """Compare optimized vs standard implementation."""
        print("\n" + "="*80)
        print("OPTIMIZED vs STANDARD GPU BENCHMARK")
        print("="*80)
        print(f"Network Size: {neurons:,} neurons")
        print(f"Iterations: {iterations}")
        print("="*80)
        
        config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
        
        results = {'neurons': neurons, 'iterations': iterations}
        
        # Standard implementation
        print("\n1. Standard Implementation...")
        try:
            brain_std = NeuroCHIMERA(config=config)
            
            if brain_std.ctx is None:
                print("  [SKIP] GPU not available")
                brain_std.release()
                return {}
            
            # Warmup
            for _ in range(3):
                brain_std.evolve(iterations=2)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(iterations):
                brain_std.evolve(iterations=1)
            std_time = time.perf_counter() - start_time
            
            std_time_per_step = std_time / iterations
            std_throughput = neurons / std_time_per_step
            
            results['standard'] = {
                'total_time': std_time,
                'time_per_step': std_time_per_step,
                'throughput': std_throughput
            }
            
            print(f"  Time per step: {std_time_per_step*1000:.2f}ms")
            print(f"  Throughput: {std_throughput/1e6:.2f}M neurons/s")
            
            brain_std.release()
        except Exception as e:
            print(f"  [ERROR] {e}")
            results['standard'] = None
        
        # Optimized implementation
        print("\n2. Optimized Implementation...")
        try:
            brain_opt = OptimizedNeuroCHIMERA(config=config)
            
            if brain_opt.ctx is None:
                print("  [SKIP] GPU not available")
                brain_opt.release()
                return results
            
            # Warmup
            for _ in range(3):
                brain_opt.evolve_optimized(iterations=2)
            
            # Benchmark
            # Ensure GPU is ready and clear any pending work
            brain_opt.ctx.finish()
            
            start_time = time.perf_counter()
            for _ in range(iterations):
                brain_opt.evolve_optimized(iterations=1)
                # Synchronize after each step to get accurate timing
                brain_opt.ctx.finish()
            opt_time = time.perf_counter() - start_time
            
            opt_time_per_step = opt_time / iterations
            opt_throughput = neurons / opt_time_per_step
            
            results['optimized'] = {
                'total_time': opt_time,
                'time_per_step': opt_time_per_step,
                'throughput': opt_throughput
            }
            
            print(f"  Time per step: {opt_time_per_step*1000:.2f}ms")
            print(f"  Throughput: {opt_throughput/1e6:.2f}M neurons/s")
            
            brain_opt.release()
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            results['optimized'] = None
        
        # Comparison
        if results.get('standard') and results.get('optimized'):
            speedup = results['standard']['time_per_step'] / results['optimized']['time_per_step']
            throughput_improvement = results['optimized']['throughput'] / results['standard']['throughput']
            
            results['speedup'] = speedup
            results['throughput_improvement'] = throughput_improvement
            
            print(f"\n{'='*80}")
            print("COMPARISON RESULTS")
            print("="*80)
            print(f"Speedup: {speedup:.2f}x")
            print(f"Throughput Improvement: {throughput_improvement:.2f}x")
            print(f"Time Reduction: {(1 - 1/speedup)*100:.1f}%")
            print("="*80)
        
        return results
    
    def benchmark_scaling(self, network_sizes: list = None):
        """Benchmark optimized implementation scaling."""
        if network_sizes is None:
            network_sizes = [262144, 1048576, 4194304]
        
        print("\n" + "="*80)
        print("OPTIMIZED GPU SCALING BENCHMARK")
        print("="*80)
        
        results = {'test_name': 'Optimized Scaling', 'data': []}
        
        for neurons in network_sizes:
            print(f"\nTesting {neurons:,} neurons...")
            
            try:
                config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
                brain = OptimizedNeuroCHIMERA(config=config)
                
                if brain.ctx is None:
                    print("  [SKIP] GPU not available")
                    brain.release()
                    continue
                
                # Warmup
                brain.evolve_optimized(iterations=2)
                
                # Benchmark
                iterations = 20
                start_time = time.perf_counter()
                for _ in range(iterations):
                    brain.evolve_optimized(iterations=1)
                total_time = time.perf_counter() - start_time
                
                time_per_step = total_time / iterations
                throughput = neurons / time_per_step
                gflops = (neurons * 25) / (time_per_step * 1e9)
                
                data_point = {
                    'neurons': neurons,
                    'texture_size': config.texture_size,
                    'time_per_step': time_per_step,
                    'throughput': throughput,
                    'gflops': gflops
                }
                
                results['data'].append(data_point)
                
                print(f"  Time per step: {time_per_step*1000:.2f}ms")
                print(f"  Throughput: {throughput/1e6:.2f}M neurons/s")
                print(f"  Compute: {gflops:.2f} GFLOPS")
                
                brain.release()
                
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def run_all(self):
        """Run all optimized benchmarks."""
        print("\n" + "="*80)
        print("OPTIMIZED GPU BENCHMARK SUITE")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nTesting optimized implementation with:")
        print("  - Compute shaders")
        print("  - Pre-allocated resources")
        print("  - GPU-only operations")
        print("  - No CPU-GPU transfers")
        
        all_results = {}
        
        # Comparison benchmark
        all_results['comparison'] = self.benchmark_comparison(neurons=1048576, iterations=50)
        
        # Scaling benchmark
        all_results['scaling'] = self.benchmark_scaling()
        
        self.results = all_results
        return all_results
    
    def save_results(self, filename: str = "optimized_gpu_benchmark_results.json"):
        """Save results to JSON."""
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


def main():
    """Main execution."""
    benchmark = OptimizedGPUBenchmark()
    results = benchmark.run_all()
    benchmark.save_results()
    
    print("\n[OK] Optimized GPU benchmark completed")


if __name__ == '__main__':
    main()

