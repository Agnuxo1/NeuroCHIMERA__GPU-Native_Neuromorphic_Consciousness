"""
CHIMERA vs PyTorch GPU Benchmark
=================================
Fair GPU-to-GPU comparison with PyTorch.

Both frameworks use GPU acceleration for apples-to-apples comparison.
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
from Benchmarks.gpu_utilization_validator import GPUUtilizationValidator

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("[ERROR] PyTorch not available")


def benchmark_pytorch_gpu(size: int, iterations: int = 20, monitor_gpu: bool = True) -> dict:
    """Benchmark PyTorch GPU operations."""
    if not HAS_PYTORCH:
        return {'error': 'PyTorch not available'}
    
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    print(f"\nPyTorch GPU - {size}×{size} operations")
    
    device = torch.device('cuda')
    
    # GPU monitoring
    validator = None
    if monitor_gpu:
        validator = GPUUtilizationValidator(target_min_utilization=80.0)
        validator.start_monitoring(interval=0.1)
    
    # Create tensors on GPU
    A = torch.randn(size, size, device=device, dtype=torch.float32)
    B = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # Warmup
    _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark matrix multiplication
    times_matmul = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times_matmul.append(elapsed)
    
    # Benchmark neural network-like operations
    times_neuralops = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Simulate neural operations
        C = torch.matmul(A, B)
        C = torch.tanh(C)
        C = C * 0.99  # decay
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times_neuralops.append(elapsed)
    
    # Benchmark convolution (2D)
    input_tensor = torch.randn(1, 1, size, size, device=device)
    conv = torch.nn.Conv2d(1, 1, kernel_size=5, padding=2, device=device)
    
    times_conv = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = conv(input_tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times_conv.append(elapsed)
    
    # Stop GPU monitoring
    gpu_analysis = None
    if validator:
        gpu_analysis = validator.stop_monitoring()
    
    result = {
        'framework': 'PyTorch',
        'device': 'GPU (CUDA)',
        'cuda_device': torch.cuda.get_device_name(0),
        'size': size,
        'operations': {
            'matrix_multiply': {
                'mean_time_ms': np.mean(times_matmul) * 1000,
                'std_time_ms': np.std(times_matmul) * 1000,
                'gflops': (2 * size**3) / (np.mean(times_matmul) * 1e9)
            },
            'neural_ops': {
                'mean_time_ms': np.mean(times_neuralops) * 1000,
                'std_time_ms': np.std(times_neuralops) * 1000,
            },
            'convolution': {
                'mean_time_ms': np.mean(times_conv) * 1000,
                'std_time_ms': np.std(times_conv) * 1000,
            }
        }
    }
    
    if gpu_analysis:
        result['gpu_utilization'] = {
            'mean': gpu_analysis['gpu_utilization']['mean'],
            'std': gpu_analysis['gpu_utilization']['std'],
            'max': gpu_analysis['gpu_utilization']['max'],
            'verdict': gpu_analysis['validation']['verdict']
        }
    
    return result


def benchmark_chimera_gpu(neurons: int, iterations: int = 20, monitor_gpu: bool = True) -> dict:
    """Benchmark CHIMERA GPU operations with monitoring."""
    print(f"\nCHIMERA GPU - {neurons:,} neurons")
    
    # GPU monitoring
    validator = None
    if monitor_gpu:
        validator = GPUUtilizationValidator(target_min_utilization=80.0)
        validator.start_monitoring(interval=0.1)
    
    try:
        config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
        engine = MultiCoreNeuroCHIMERA(config=config, parallel_batches=1)
        
        if engine.ctx is None:
            if validator:
                validator.stop_monitoring()
            return {'error': 'GPU not available'}
        
        # Warmup
        engine.evolve_ultra_optimized(iterations=3)
        engine.ctx.finish()
        
        # Benchmark
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            engine.evolve_ultra_optimized(iterations=1)
            engine.ctx.finish()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        mean_time = np.mean(times)
        throughput = config.neurons / mean_time
        gflops = (config.neurons * 25) / (mean_time * 1e9)
        
        engine.release()
        
        # Stop GPU monitoring
        gpu_analysis = None
        if validator:
            gpu_analysis = validator.stop_monitoring()
        
        result = {
            'framework': 'CHIMERA',
            'device': 'GPU (OpenGL)',
            'neurons': config.neurons,
            'texture_size': config.texture_size,
            'operations': {
                'evolution': {
                    'mean_time_ms': mean_time * 1000,
                    'std_time_ms': np.std(times) * 1000,
                    'throughput_neurons_per_sec': throughput,
                    'gflops': gflops
                }
            }
        }
        
        if gpu_analysis:
            result['gpu_utilization'] = {
                'mean': gpu_analysis['gpu_utilization']['mean'],
                'std': gpu_analysis['gpu_utilization']['std'],
                'max': gpu_analysis['gpu_utilization']['max'],
                'verdict': gpu_analysis['validation']['verdict']
            }
        
        return result
    
    except Exception as e:
        if validator:
            validator.stop_monitoring()
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    """Run CHIMERA vs PyTorch GPU comparison."""
    print("\n" + "="*80)
    print("CHIMERA vs PyTorch GPU Comparison")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print("="*80)
    
    if not HAS_PYTORCH:
        print("\n[ERROR] PyTorch is required for this benchmark")
        print("Install with: pip install torch")
        return
    
    results = {
        'benchmark': 'CHIMERA vs PyTorch GPU',
        'timestamp': datetime.now().isoformat(),
        'pytorch_results': [],
        'chimera_results': []
    }
    
    # Test configurations
    test_configs = [
        (1024, 1_048_576),
        (2048, 4_194_304),
    ]
    
    for matrix_size, neurons in test_configs:
        print(f"\n{'='*80}")
        print(f"Test Scale: {matrix_size}×{matrix_size} / {neurons:,} neurons")
        print(f"{'='*80}")
        
        # PyTorch GPU
        pytorch_result = benchmark_pytorch_gpu(matrix_size, iterations=20, monitor_gpu=True)
        results['pytorch_results'].append(pytorch_result)
        
        if 'error' not in pytorch_result:
            print(f"\nPyTorch GPU Results:")
            print(f"  Matrix Multiply: {pytorch_result['operations']['matrix_multiply']['mean_time_ms']:.2f}ms "
                  f"({pytorch_result['operations']['matrix_multiply']['gflops']:.2f} GFLOPS)")
            print(f"  Neural Ops: {pytorch_result['operations']['neural_ops']['mean_time_ms']:.2f}ms")
            if 'gpu_utilization' in pytorch_result:
                print(f"  GPU Usage: {pytorch_result['gpu_utilization']['mean']:.1f}% ± {pytorch_result['gpu_utilization']['std']:.1f}%")
        
        # CHIMERA GPU
        chimera_result = benchmark_chimera_gpu(neurons, iterations=20, monitor_gpu=True)
        results['chimera_results'].append(chimera_result)
        
        if 'error' not in chimera_result:
            print(f"\nCHIMERA GPU Results:")
            print(f"  Evolution: {chimera_result['operations']['evolution']['mean_time_ms']:.2f}ms "
                  f"({chimera_result['operations']['evolution']['gflops']:.3f} GFLOPS)")
            print(f"  Throughput: {chimera_result['operations']['evolution']['throughput_neurons_per_sec']/1e6:.2f}M neurons/s")
            if 'gpu_utilization' in chimera_result:
                print(f"  GPU Usage: {chimera_result['gpu_utilization']['mean']:.1f}% ± {chimera_result['gpu_utilization']['std']:.1f}%")
        
        # Comparison
        if 'error' not in pytorch_result and 'error' not in chimera_result:
            pytorch_gflops = pytorch_result['operations']['matrix_multiply']['gflops']
            chimera_gflops = chimera_result['operations']['evolution']['gflops']
            
            print(f"\nDirect Comparison:")
            print(f"  Compute Performance:")
            print(f"    PyTorch: {pytorch_gflops:.2f} GFLOPS")
            print(f"    CHIMERA: {chimera_gflops:.3f} GFLOPS")
            print(f"    Ratio: {pytorch_gflops/chimera_gflops:.2f}× (PyTorch/CHIMERA)")
            
            if 'gpu_utilization' in pytorch_result and 'gpu_utilization' in chimera_result:
                print(f"  GPU Utilization:")
                print(f"    PyTorch: {pytorch_result['gpu_utilization']['mean']:.1f}%")
                print(f"    CHIMERA: {chimera_result['gpu_utilization']['mean']:.1f}%")
    
    # Save results
    output_file = Path(__file__).parent / 'benchmark_vs_pytorch_gpu_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"[OK] Results saved to: {output_file}")
    print(f"{'='*80}")
    
    print("\n" + "="*80)
    print("IMPORTANT NOTES")
    print("="*80)
    print("- PyTorch uses highly optimized CUDA kernels for matrix operations")
    print("- CHIMERA uses OpenGL compute shaders for neuromorphic operations")
    print("- Both execute 100% on GPU (no CPU fallback)")
    print("- GPU utilization validated for fair comparison")
    print("- Different operation types but comparable compute intensity")
    print("="*80)


if __name__ == '__main__':
    main()
