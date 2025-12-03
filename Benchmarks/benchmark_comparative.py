"""
Comparative Benchmarks: CHIMERA vs Alternatives
===============================================

Compare NeuroCHIMERA performance against:
- PyTorch
- Standard neural networks
- GPU utilization
"""

import time
import numpy as np
import sys
import json
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import NeuroCHIMERA, NeuroCHIMERAConfig

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("Warning: PyTorch not available. PyTorch comparisons will be skipped.")


class ComparativeBenchmark:
    """Comparative benchmarking suite."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_matrix_operations(self, size: int = 2048) -> Dict:
        """Compare matrix operations: CHIMERA vs PyTorch."""
        print("\n" + "="*80)
        print(f"MATRIX OPERATIONS BENCHMARK ({size}×{size})")
        print("="*80)
        
        results = {'test_name': 'Matrix Operations', 'size': size}
        
        # Generate test data
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # PyTorch
        if HAS_PYTORCH:
            print("\nPyTorch...")
            a_torch = torch.from_numpy(a).cuda() if torch.cuda.is_available() else torch.from_numpy(a)
            b_torch = torch.from_numpy(b).cuda() if torch.cuda.is_available() else torch.from_numpy(b)
            
            # Warmup
            _ = torch.mm(a_torch, b_torch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            iterations = 100
            start_time = time.perf_counter()
            for _ in range(iterations):
                _ = torch.mm(a_torch, b_torch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pytorch_time = time.perf_counter() - start_time
            
            pytorch_time_per_op = pytorch_time / iterations
            
            results['pytorch'] = {
                'total_time': pytorch_time,
                'time_per_op': pytorch_time_per_op,
                'ops_per_second': iterations / pytorch_time
            }
            
            print(f"  Time per operation: {pytorch_time_per_op*1000:.2f}ms")
        else:
            results['pytorch'] = None
            print("\nPyTorch not available")
        
        # CHIMERA (using evolution as proxy for matrix-like operations)
        print("\nNeuroCHIMERA...")
        config = NeuroCHIMERAConfig(neurons=size*size, use_hns=True)
        brain = NeuroCHIMERA(config=config)
        
        # Warmup
        brain.evolve(iterations=2)
        
        # Benchmark
        iterations = 100
        start_time = time.perf_counter()
        for _ in range(iterations):
            brain.evolve(iterations=5)
        chimera_time = time.perf_counter() - start_time
        
        chimera_time_per_op = chimera_time / iterations
        
        results['chimera'] = {
            'total_time': chimera_time,
            'time_per_op': chimera_time_per_op,
            'ops_per_second': iterations / chimera_time
        }
        
        print(f"  Time per operation: {chimera_time_per_op*1000:.2f}ms")
        
        # Comparison
        if HAS_PYTORCH and results['pytorch']:
            speedup = pytorch_time_per_op / chimera_time_per_op
            results['speedup'] = speedup
            print(f"\nSpeedup: {speedup:.2f}x")
        
        brain.release()
        
        return results
    
    def benchmark_memory_footprint(self, neurons: int = 1048576) -> Dict:
        """Compare memory footprint: CHIMERA vs PyTorch."""
        print("\n" + "="*80)
        print(f"MEMORY FOOTPRINT BENCHMARK ({neurons:,} neurons)")
        print("="*80)
        
        results = {'test_name': 'Memory Footprint', 'neurons': neurons}
        
        # PyTorch
        if HAS_PYTORCH:
            print("\nPyTorch...")
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create equivalent PyTorch network
            layer_size = int(np.sqrt(neurons))
            model = torch.nn.Sequential(
                torch.nn.Linear(layer_size, layer_size),
                torch.nn.ReLU(),
                torch.nn.Linear(layer_size, layer_size)
            )
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            pytorch_memory = process.memory_info().rss / 1024 / 1024 - baseline_memory
            
            results['pytorch'] = {
                'memory_mb': pytorch_memory,
                'memory_per_neuron_kb': pytorch_memory * 1024 / neurons
            }
            
            print(f"  Memory: {pytorch_memory:.1f} MB")
            print(f"  Per neuron: {pytorch_memory * 1024 / neurons:.3f} KB")
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            results['pytorch'] = None
            print("\nPyTorch not available")
        
        # CHIMERA
        print("\nNeuroCHIMERA...")
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
        brain = NeuroCHIMERA(config=config)
        
        chimera_memory = process.memory_info().rss / 1024 / 1024 - baseline_memory
        
        results['chimera'] = {
            'memory_mb': chimera_memory,
            'memory_per_neuron_kb': chimera_memory * 1024 / neurons
        }
        
        print(f"  Memory: {chimera_memory:.1f} MB")
        print(f"  Per neuron: {chimera_memory * 1024 / neurons:.3f} KB")
        
        # Comparison
        if HAS_PYTORCH and results['pytorch']:
            reduction = (1 - chimera_memory / pytorch_memory) * 100
            results['memory_reduction_percent'] = reduction
            print(f"\nMemory reduction: {reduction:.1f}%")
        
        brain.release()
        
        return results
    
    def benchmark_synaptic_updates(self, num_synapses: int = 1000000) -> Dict:
        """Compare synaptic update performance."""
        print("\n" + "="*80)
        print(f"SYNAPTIC UPDATES BENCHMARK ({num_synapses:,} synapses)")
        print("="*80)
        
        results = {'test_name': 'Synaptic Updates', 'num_synapses': num_synapses}
        
        # PyTorch
        if HAS_PYTORCH:
            print("\nPyTorch...")
            weights = torch.randn(num_synapses, dtype=torch.float32)
            if torch.cuda.is_available():
                weights = weights.cuda()
            
            activations = torch.randn(num_synapses, dtype=torch.float32)
            if torch.cuda.is_available():
                activations = activations.cuda()
            
            learning_rate = 0.01
            
            # Warmup
            weights = weights + learning_rate * activations
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            iterations = 1000
            start_time = time.perf_counter()
            for _ in range(iterations):
                weights = weights + learning_rate * activations
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pytorch_time = time.perf_counter() - start_time
            
            pytorch_time_per_update = pytorch_time / iterations
            
            results['pytorch'] = {
                'total_time': pytorch_time,
                'time_per_update': pytorch_time_per_update,
                'updates_per_second': iterations / pytorch_time
            }
            
            print(f"  Time per update: {pytorch_time_per_update*1000:.3f}ms")
            print(f"  Updates/s: {iterations / pytorch_time:,.0f}")
        else:
            results['pytorch'] = None
            print("\nPyTorch not available")
        
        # CHIMERA (using learning as proxy)
        print("\nNeuroCHIMERA...")
        neurons = int(np.sqrt(num_synapses))
        config = NeuroCHIMERAConfig(neurons=neurons*neurons, use_hns=True)
        brain = NeuroCHIMERA(config=config)
        
        # Warmup
        brain.learn(learning_rate=0.01)
        
        # Benchmark
        iterations = 1000
        start_time = time.perf_counter()
        for _ in range(iterations):
            brain.learn(learning_rate=0.01)
        chimera_time = time.perf_counter() - start_time
        
        chimera_time_per_update = chimera_time / iterations
        
        results['chimera'] = {
            'total_time': chimera_time,
            'time_per_update': chimera_time_per_update,
            'updates_per_second': iterations / chimera_time
        }
        
        print(f"  Time per update: {chimera_time_per_update*1000:.3f}ms")
        print(f"  Updates/s: {iterations / chimera_time:,.0f}")
        
        # Comparison
        if HAS_PYTORCH and results['pytorch']:
            speedup = pytorch_time_per_update / chimera_time_per_update
            results['speedup'] = speedup
            print(f"\nSpeedup: {speedup:.2f}x")
        
        brain.release()
        
        return results
    
    def run_all(self) -> Dict:
        """Run all comparative benchmarks."""
        print("\n" + "="*80)
        print("COMPARATIVE BENCHMARK SUITE: CHIMERA vs ALTERNATIVES")
        print("="*80)
        print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = {}
        
        all_results['matrix_operations'] = self.benchmark_matrix_operations(size=2048)
        all_results['memory_footprint'] = self.benchmark_memory_footprint(neurons=1048576)
        all_results['synaptic_updates'] = self.benchmark_synaptic_updates(num_synapses=1000000)
        
        self.results = all_results
        return all_results
    
    def save_results(self, filename: str = "comparative_benchmark_results.json"):
        """Save results to JSON file."""
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


def main():
    """Main benchmark execution."""
    benchmark = ComparativeBenchmark()
    results = benchmark.run_all()
    benchmark.save_results()
    
    print("\n[OK] Comparative benchmark completed")


if __name__ == '__main__':
    main()

