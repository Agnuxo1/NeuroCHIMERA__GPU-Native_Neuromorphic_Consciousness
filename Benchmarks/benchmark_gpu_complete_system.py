"""
Complete GPU Benchmark: NeuroCHIMERA 100% on GPU
================================================

Benchmark the complete NeuroCHIMERA system executing 100% on GPU using GLSL shaders.
All operations (evolution, learning, metrics) run entirely on GPU.
"""

import time
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import NeuroCHIMERA, NeuroCHIMERAConfig
from consciousness_monitor import ConsciousnessMonitor


class GPUCompleteSystemBenchmark:
    """Complete system benchmark running 100% on GPU."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_gpu_evolution_speed(self, network_sizes: List[int] = None) -> Dict:
        """Benchmark GPU evolution speed for different network sizes."""
        print("\n" + "="*80)
        print("GPU EVOLUTION SPEED BENCHMARK (100% GPU)")
        print("="*80)
        
        if network_sizes is None:
            network_sizes = [65536, 262144, 1048576]
        
        results = {'test_name': 'GPU Evolution Speed', 'configurations': []}
        
        for neurons in network_sizes:
            print(f"\nTesting {neurons:,} neurons (100% GPU)...")
            
            config = NeuroCHIMERAConfig(
                neurons=neurons,
                default_iterations=10,
                use_hns=True  # Enable HNS for GPU
            )
            
            brain = NeuroCHIMERA(config=config)
            
            # Verify GPU is being used
            if brain.ctx is None:
                print("  [WARNING] GPU not available, skipping...")
                brain.release()
                continue
            
            print(f"  GPU: {brain.ctx.info.get('GL_RENDERER', 'Unknown')}")
            
            # Warmup (GPU needs warmup for accurate timing)
            for _ in range(3):
                brain.evolve(iterations=2)
            
            # Benchmark evolution (all on GPU)
            iterations = 50
            start_time = time.perf_counter()
            for _ in range(iterations):
                brain.evolve(iterations=5)
            total_time = time.perf_counter() - start_time
            
            time_per_step = total_time / iterations
            neurons_per_second = neurons / time_per_step
            gflops = (neurons * 25) / (time_per_step * 1e9)  # 25 ops per neuron (5x5 neighborhood)
            
            config_result = {
                'neurons': neurons,
                'texture_size': config.texture_size,
                'total_time': total_time,
                'iterations': iterations,
                'time_per_step': time_per_step,
                'neurons_per_second': neurons_per_second,
                'gflops': gflops
            }
            
            results['configurations'].append(config_result)
            
            print(f"  Time per step: {time_per_step*1000:.2f}ms")
            print(f"  Throughput: {neurons_per_second/1e6:.2f}M neurons/s")
            print(f"  Compute: {gflops:.2f} GFLOPS")
            
            brain.release()
        
        return results
    
    def benchmark_gpu_memory_bandwidth(self) -> Dict:
        """Benchmark GPU memory bandwidth usage."""
        print("\n" + "="*80)
        print("GPU MEMORY BANDWIDTH BENCHMARK")
        print("="*80)
        
        config = NeuroCHIMERAConfig(neurons=1048576, use_hns=True)
        brain = NeuroCHIMERA(config=config)
        
        if brain.ctx is None:
            print("[WARNING] GPU not available")
            brain.release()
            return {}
        
        results = {'test_name': 'GPU Memory Bandwidth'}
        
        # Calculate texture sizes
        size = config.texture_size
        bytes_per_pixel = 16  # RGBA float32 = 4 bytes * 4 channels
        
        # Main textures
        neural_texture = size * size * bytes_per_pixel
        connectivity_texture = size * size * bytes_per_pixel
        spatial_texture = size * size * bytes_per_pixel
        memory_texture = config.memory_texture_size * config.memory_texture_size * bytes_per_pixel
        
        total_texture_memory = (neural_texture + connectivity_texture + 
                               spatial_texture + memory_texture) / 1024 / 1024  # MB
        
        # Benchmark read/write operations
        iterations = 100
        start_time = time.perf_counter()
        for _ in range(iterations):
            brain.evolve(iterations=1)
        total_time = time.perf_counter() - start_time
        
        # Each evolution step reads and writes textures
        # Read: neural_state, connectivity, memory
        # Write: neural_state, spatial_features
        bytes_per_step = (neural_texture * 2 + connectivity_texture + 
                         memory_texture + spatial_texture)  # Read + Write
        
        bandwidth_gbps = (bytes_per_step * iterations) / (total_time * 1e9)
        
        results['texture_memory_mb'] = total_texture_memory
        results['bytes_per_step'] = bytes_per_step
        results['bandwidth_gbps'] = bandwidth_gbps
        results['time_per_step'] = total_time / iterations
        
        print(f"\nTexture Memory: {total_texture_memory:.1f} MB")
        print(f"Bytes per step: {bytes_per_step / 1024 / 1024:.1f} MB")
        print(f"Memory Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"Time per step: {total_time/iterations*1000:.2f}ms")
        
        brain.release()
        
        return results
    
    def benchmark_gpu_scaling(self) -> Dict:
        """Test GPU performance scaling with network size."""
        print("\n" + "="*80)
        print("GPU SCALING BENCHMARK")
        print("="*80)
        
        results = {'test_name': 'GPU Scaling', 'data': []}
        
        network_sizes = [16384, 65536, 262144, 1048576]
        
        for neurons in network_sizes:
            print(f"\nTesting {neurons:,} neurons...")
            
            config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
            brain = NeuroCHIMERA(config=config)
            
            if brain.ctx is None:
                print("  [WARNING] GPU not available")
                brain.release()
                continue
            
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
            efficiency = neurons_per_second / neurons  # Higher is better
            
            data_point = {
                'neurons': neurons,
                'texture_size': config.texture_size,
                'time_per_step': time_per_step,
                'neurons_per_second': neurons_per_second,
                'efficiency': efficiency
            }
            
            results['data'].append(data_point)
            
            print(f"  Time per step: {time_per_step*1000:.2f}ms")
            print(f"  Throughput: {neurons_per_second/1e6:.2f}M neurons/s")
            print(f"  Efficiency: {efficiency:.2e}")
            
            brain.release()
        
        return results
    
    def benchmark_gpu_vs_cpu(self) -> Dict:
        """Compare GPU vs CPU performance."""
        print("\n" + "="*80)
        print("GPU vs CPU PERFORMANCE COMPARISON")
        print("="*80)
        
        config = NeuroCHIMERAConfig(neurons=262144, use_hns=True)
        
        results = {'test_name': 'GPU vs CPU', 'neurons': 262144}
        
        # GPU benchmark
        print("\nGPU Benchmark...")
        brain_gpu = NeuroCHIMERA(config=config)
        
        if brain_gpu.ctx is not None:
            # Warmup
            for _ in range(3):
                brain_gpu.evolve(iterations=2)
            
            iterations = 50
            start_time = time.perf_counter()
            for _ in range(iterations):
                brain_gpu.evolve(iterations=5)
            gpu_time = time.perf_counter() - start_time
            
            gpu_time_per_step = gpu_time / iterations
            
            results['gpu'] = {
                'total_time': gpu_time,
                'time_per_step': gpu_time_per_step,
                'neurons_per_second': config.neurons / gpu_time_per_step
            }
            
            print(f"  Time per step: {gpu_time_per_step*1000:.2f}ms")
            print(f"  Throughput: {config.neurons / gpu_time_per_step / 1e6:.2f}M neurons/s")
            
            brain_gpu.release()
        else:
            print("  [WARNING] GPU not available")
            results['gpu'] = None
        
        # CPU benchmark (force CPU mode by disabling GPU)
        print("\nCPU Benchmark...")
        # Note: We can't easily force CPU mode, but we can estimate
        # based on the fact that CPU would be much slower
        # For now, we'll just report GPU results
        
        if results['gpu']:
            results['speedup'] = "N/A (CPU mode not available for comparison)"
            print("\nNote: CPU comparison requires CPU-only mode implementation")
        
        return results
    
    def benchmark_gpu_throughput(self) -> Dict:
        """Benchmark GPU operations throughput."""
        print("\n" + "="*80)
        print("GPU OPERATIONS THROUGHPUT")
        print("="*80)
        
        config = NeuroCHIMERAConfig(neurons=1048576, use_hns=True)
        brain = NeuroCHIMERA(config=config)
        
        if brain.ctx is None:
            print("[WARNING] GPU not available")
            brain.release()
            return {}
        
        results = {'test_name': 'GPU Throughput', 'operations': {}}
        
        # Evolution throughput
        print("\nEvolution throughput (GPU)...")
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
        
        # Learning throughput (GPU)
        print("\nLearning throughput (GPU)...")
        start_time = time.perf_counter()
        for _ in range(iterations):
            brain.learn(learning_rate=0.001)
        total_time = time.perf_counter() - start_time
        
        learning_per_second = iterations / total_time
        
        results['operations']['learning'] = {
            'updates_per_second': learning_per_second,
            'time_per_update': total_time / iterations
        }
        
        print(f"  Updates/s: {learning_per_second:,.0f}")
        
        brain.release()
        
        return results
    
    def run_all(self) -> Dict:
        """Run all GPU benchmarks."""
        print("\n" + "="*80)
        print("COMPLETE GPU BENCHMARK SUITE: NeuroCHIMERA 100% on GPU")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nAll operations execute 100% on GPU using GLSL shaders")
        print("No CPU computation involved in evolution/learning operations")
        
        all_results = {}
        
        all_results['evolution_speed'] = self.benchmark_gpu_evolution_speed()
        all_results['memory_bandwidth'] = self.benchmark_gpu_memory_bandwidth()
        all_results['scaling'] = self.benchmark_gpu_scaling()
        all_results['throughput'] = self.benchmark_gpu_throughput()
        all_results['gpu_vs_cpu'] = self.benchmark_gpu_vs_cpu()
        
        self.results = all_results
        return all_results
    
    def save_results(self, filename: str = "gpu_complete_system_benchmark_results.json"):
        """Save results to JSON file."""
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


def main():
    """Main benchmark execution."""
    benchmark = GPUCompleteSystemBenchmark()
    results = benchmark.run_all()
    benchmark.save_results()
    
    # Summary
    print("\n" + "="*80)
    print("GPU BENCHMARK SUMMARY")
    print("="*80)
    
    if 'evolution_speed' in results and results['evolution_speed'].get('configurations'):
        configs = results['evolution_speed']['configurations']
        if configs:
            largest = max(configs, key=lambda x: x.get('neurons', 0))
            print(f"\nLargest Network ({largest.get('neurons', 0):,} neurons):")
            print(f"  Time per step: {largest.get('time_per_step', 0)*1000:.2f}ms")
            print(f"  Throughput: {largest.get('neurons_per_second', 0)/1e6:.2f}M neurons/s")
            print(f"  Compute: {largest.get('gflops', 0):.2f} GFLOPS")
    
    if 'memory_bandwidth' in results:
        mem = results['memory_bandwidth']
        print(f"\nMemory Bandwidth: {mem.get('bandwidth_gbps', 0):.2f} GB/s")
    
    print("\n[OK] Complete GPU benchmark completed")


if __name__ == '__main__':
    main()

