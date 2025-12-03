"""
NeuroCHIMERA System Performance Benchmarks
==========================================

System-level performance tests:
- Evolution speed
- Memory efficiency
- Scalability
- Throughput
"""

import time
import numpy as np
import sys
import json
import psutil
import os
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import NeuroCHIMERA, NeuroCHIMERAConfig


class SystemBenchmark:
    """System-level performance benchmarking."""
    
    def __init__(self):
        self.results = {}
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_evolution_speed(self, network_sizes: List[int] = None) -> Dict:
        """Benchmark evolution speed for different network sizes."""
        print("\n" + "="*80)
        print("EVOLUTION SPEED BENCHMARK")
        print("="*80)
        
        if network_sizes is None:
            network_sizes = [65536, 262144, 1048576]  # 256², 512², 1024²
        
        results = {'test_name': 'Evolution Speed', 'configurations': []}
        
        for neurons in network_sizes:
            print(f"\nTesting {neurons:,} neurons...")
            
            config = NeuroCHIMERAConfig(
                neurons=neurons,
                default_iterations=10,
                use_hns=True
            )
            
            brain = NeuroCHIMERA(config=config)
            
            # CERTIFICATION: Enforce GPU usage
            if not brain.use_compute_shaders:
                print("  [ERROR] Not running on GPU (Compute Shaders). Aborting benchmark.")
                brain.release()
                return {'error': 'GPU required'}
            
            print(f"  [CERTIFIED] Running on GPU: {brain.ctx.info['GL_RENDERER']}")
            
            # Warmup
            brain.evolve(iterations=2)
            
            # Benchmark
            iterations = 20
            start_time = time.perf_counter()
            for _ in range(iterations):
                brain.evolve(iterations=5)
            total_time = time.perf_counter() - start_time
            
            time_per_step = total_time / iterations
            neurons_per_second = neurons / time_per_step
            
            config_result = {
                'neurons': neurons,
                'texture_size': config.texture_size,
                'total_time': total_time,
                'iterations': iterations,
                'time_per_step': time_per_step,
                'neurons_per_second': neurons_per_second,
                'memory_mb': self.get_memory_usage()
            }
            
            results['configurations'].append(config_result)
            
            print(f"  Time per step: {time_per_step*1000:.2f}ms")
            print(f"  Throughput: {neurons_per_second/1e6:.2f}M neurons/s")
            print(f"  Memory: {config_result['memory_mb']:.1f} MB")
            
            brain.release()
        
        return results
    
    def benchmark_memory_efficiency(self) -> Dict:
        """Benchmark memory efficiency."""
        print("\n" + "="*80)
        print("MEMORY EFFICIENCY BENCHMARK")
        print("="*80)
        
        results = {'test_name': 'Memory Efficiency', 'configurations': []}
        
        network_sizes = [65536, 262144, 1048576]
        
        for neurons in network_sizes:
            print(f"\nTesting {neurons:,} neurons...")
            
            # Measure baseline memory
            baseline_memory = self.get_memory_usage()
            
            config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
            brain = NeuroCHIMERA(config=config)
            
            # Measure after initialization
            init_memory = self.get_memory_usage()
            memory_used = init_memory - baseline_memory
            
            # Calculate theoretical memory (textures)
            texture_size = config.texture_size
            # RGBA float32 = 4 bytes per channel * 4 channels = 16 bytes per pixel
            bytes_per_pixel = 16
            neural_texture = texture_size * texture_size * bytes_per_pixel
            connectivity_texture = texture_size * texture_size * bytes_per_pixel
            spatial_texture = texture_size * texture_size * bytes_per_pixel
            memory_texture = config.memory_texture_size * config.memory_texture_size * bytes_per_pixel
            
            total_texture_memory = (neural_texture + connectivity_texture + 
                                   spatial_texture + memory_texture) / 1024 / 1024  # MB
            
            config_result = {
                'neurons': neurons,
                'texture_size': texture_size,
                'memory_used_mb': memory_used,
                'theoretical_texture_mb': total_texture_memory,
                'efficiency': total_texture_memory / memory_used if memory_used > 0 else 0
            }
            
            results['configurations'].append(config_result)
            
            print(f"  Memory used: {memory_used:.1f} MB")
            print(f"  Theoretical texture memory: {total_texture_memory:.1f} MB")
            print(f"  Efficiency: {config_result['efficiency']:.2f}x")
            
            brain.release()
        
        return results
    
    def benchmark_scalability(self) -> Dict:
        """Test performance scaling with network size."""
        print("\n" + "="*80)
        print("SCALABILITY BENCHMARK")
        print("="*80)
        
        results = {'test_name': 'Scalability', 'data': []}
        
        network_sizes = [16384, 65536, 262144, 1048576]  # 128² to 1024²
        
        for neurons in network_sizes:
            print(f"\nTesting {neurons:,} neurons...")
            
            config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
            brain = NeuroCHIMERA(config=config)
            
            # Benchmark single evolution step
            start_time = time.perf_counter()
            brain.evolve(iterations=10)
            step_time = time.perf_counter() - start_time
            
            # Calculate metrics
            time_per_neuron = step_time / neurons
            neurons_per_second = neurons / step_time
            
            data_point = {
                'neurons': neurons,
                'texture_size': config.texture_size,
                'step_time': step_time,
                'time_per_neuron': time_per_neuron,
                'neurons_per_second': neurons_per_second,
                'memory_mb': self.get_memory_usage()
            }
            
            results['data'].append(data_point)
            
            print(f"  Step time: {step_time*1000:.2f}ms")
            print(f"  Time per neuron: {time_per_neuron*1e6:.3f}μs")
            print(f"  Throughput: {neurons_per_second/1e6:.2f}M neurons/s")
            
            brain.release()
        
        return results
    
    def benchmark_throughput(self) -> Dict:
        """Benchmark operations throughput."""
        print("\n" + "="*80)
        print("THROUGHPUT BENCHMARK")
        print("="*80)
        
        config = NeuroCHIMERAConfig(neurons=1048576, use_hns=True)
        brain = NeuroCHIMERA(config=config)
        
        results = {'test_name': 'Throughput', 'operations': {}}
        
        # Evolution throughput
        print("\nEvolution throughput...")
        iterations = 100
        start_time = time.perf_counter()
        for _ in range(iterations):
            brain.evolve(iterations=5)
        total_time = time.perf_counter() - start_time
        
        evolutions_per_second = iterations / total_time
        neurons_per_second = config.neurons * evolutions_per_second
        
        results['operations']['evolution'] = {
            'evolutions_per_second': evolutions_per_second,
            'neurons_per_second': neurons_per_second,
            'time_per_evolution': total_time / iterations
        }
        
        print(f"  Evolutions/s: {evolutions_per_second:.2f}")
        print(f"  Neurons/s: {neurons_per_second/1e6:.2f}M")
        
        # Learning throughput
        print("\nLearning throughput...")
        start_time = time.perf_counter()
        for _ in range(iterations):
            brain.learn(learning_rate=0.001)
        total_time = time.perf_counter() - start_time
        
        learning_per_second = iterations / total_time
        
        results['operations']['learning'] = {
            'updates_per_second': learning_per_second,
            'time_per_update': total_time / iterations
        }
        
        print(f"  Updates/s: {learning_per_second:.2f}")
        
        # Metrics calculation throughput
        print("\nMetrics calculation throughput...")
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = brain.get_metrics()
        total_time = time.perf_counter() - start_time
        
        metrics_per_second = iterations / total_time
        
        results['operations']['metrics'] = {
            'calculations_per_second': metrics_per_second,
            'time_per_calculation': total_time / iterations
        }
        
        print(f"  Calculations/s: {metrics_per_second:.2f}")
        
        brain.release()
        
        return results
    
    def run_all(self) -> Dict:
        """Run all system benchmarks."""
        print("\n" + "="*80)
        print("NEUROCHIMERA SYSTEM BENCHMARK SUITE")
        print("="*80)
        print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = {}
        
        all_results['evolution_speed'] = self.benchmark_evolution_speed()
        all_results['memory_efficiency'] = self.benchmark_memory_efficiency()
        all_results['scalability'] = self.benchmark_scalability()
        all_results['throughput'] = self.benchmark_throughput()
        
        self.results = all_results
        return all_results
    
    def save_results(self, filename: str = "system_benchmark_results.json"):
        """Save results to JSON file."""
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


def main():
    """Main benchmark execution."""
    benchmark = SystemBenchmark()
    results = benchmark.run_all()
    benchmark.save_results()
    
    print("\n[OK] System benchmark completed")


if __name__ == '__main__':
    main()

