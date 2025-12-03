"""
Complete System Benchmark
=========================

Full system performance benchmark combining all aspects:
- Evolution speed
- Memory efficiency
- Consciousness parameter tracking
- Throughput
"""

import time
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from engine import NeuroCHIMERA, NeuroCHIMERAConfig
from consciousness_monitor import ConsciousnessMonitor


def benchmark_complete_system(
    neurons: int = 1048576,
    epochs: int = 100,
    output_dir: Path = None
):
    """
    Run complete system benchmark.
    
    Args:
        neurons: Number of neurons
        epochs: Number of epochs to benchmark
        output_dir: Directory to save results
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("COMPLETE SYSTEM BENCHMARK")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Neurons: {neurons:,}")
    print(f"  Epochs: {epochs:,}")
    print("=" * 80)
    print()
    
    config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
    brain = NeuroCHIMERA(config=config)
    monitor = ConsciousnessMonitor(brain)
    
    results = {
        'configuration': {
            'neurons': neurons,
            'texture_size': config.texture_size,
            'epochs': epochs
        },
        'timings': {},
        'metrics': [],
        'memory': {}
    }
    
    # Benchmark evolution
    print("Benchmarking evolution...")
    evolution_times = []
    for epoch in range(epochs):
        start = time.perf_counter()
        brain.evolve(iterations=10)
        evolution_times.append(time.perf_counter() - start)
    
    results['timings']['evolution'] = {
        'total_time': sum(evolution_times),
        'mean_time': sum(evolution_times) / len(evolution_times),
        'min_time': min(evolution_times),
        'max_time': max(evolution_times),
        'std_time': (sum((t - sum(evolution_times)/len(evolution_times))**2 for t in evolution_times) / len(evolution_times))**0.5
    }
    
    print(f"  Mean time per evolution: {results['timings']['evolution']['mean_time']*1000:.2f}ms")
    
    # Benchmark learning
    print("Benchmarking learning...")
    learning_times = []
    for _ in range(epochs):
        start = time.perf_counter()
        brain.learn(learning_rate=0.01)
        learning_times.append(time.perf_counter() - start)
    
    results['timings']['learning'] = {
        'total_time': sum(learning_times),
        'mean_time': sum(learning_times) / len(learning_times),
        'min_time': min(learning_times),
        'max_time': max(learning_times)
    }
    
    print(f"  Mean time per learning: {results['timings']['learning']['mean_time']*1000:.2f}ms")
    
    # Benchmark metrics calculation
    print("Benchmarking metrics calculation...")
    metrics_times = []
    for _ in range(epochs):
        start = time.perf_counter()
        metrics = monitor.measure()
        metrics_times.append(time.perf_counter() - start)
        results['metrics'].append(metrics.to_dict())
    
    results['timings']['metrics'] = {
        'total_time': sum(metrics_times),
        'mean_time': sum(metrics_times) / len(metrics_times),
        'min_time': min(metrics_times),
        'max_time': max(metrics_times)
    }
    
    print(f"  Mean time per metrics: {results['timings']['metrics']['mean_time']*1000:.2f}ms")
    
    # Memory usage
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        results['memory']['rss_mb'] = process.memory_info().rss / 1024 / 1024
        results['memory']['vms_mb'] = process.memory_info().vms / 1024 / 1024
        print(f"\nMemory usage: {results['memory']['rss_mb']:.1f} MB RSS")
    except ImportError:
        print("\npsutil not available for memory measurement")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"complete_system_benchmark_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    total_time = (
        results['timings']['evolution']['total_time'] +
        results['timings']['learning']['total_time'] +
        results['timings']['metrics']['total_time']
    )
    print(f"Total time: {total_time:.2f}s")
    print(f"Time per epoch: {total_time/epochs*1000:.2f}ms")
    print(f"Throughput: {epochs/total_time:.2f} epochs/s")
    print("=" * 80)
    
    brain.release()
    
    return results


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete system benchmark')
    parser.add_argument('--neurons', type=int, default=1048576, help='Number of neurons')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    benchmark_complete_system(
        neurons=args.neurons,
        epochs=args.epochs,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()

