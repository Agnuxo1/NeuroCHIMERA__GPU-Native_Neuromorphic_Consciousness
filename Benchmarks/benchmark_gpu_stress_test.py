"""
GPU Stress Test: Maximum Network Size Benchmark
================================================

Push RTX 3090 to its limits to find the maximum number of neurons
that can be processed. This benchmark tests progressively larger
networks until GPU memory or performance limits are reached.

Target: Find maximum viable network size for RTX 3090
"""

import time
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import NeuroCHIMERA, NeuroCHIMERAConfig


class GPUStressTest:
    """Stress test to find maximum GPU capacity."""
    
    def __init__(self):
        self.results = {}
        self.max_successful_size = 0
        self.failure_point = None
    
    def test_network_size(self, neurons: int, iterations: int = 5) -> Tuple[bool, Dict]:
        """
        Test if a network size can be created and run.
        
        Returns:
            (success, metrics_dict)
        """
        print(f"\n{'='*80}")
        print(f"Testing {neurons:,} neurons ({int(np.sqrt(neurons))}×{int(np.sqrt(neurons))} texture)")
        print(f"{'='*80}")
        
        try:
            config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
            brain = NeuroCHIMERA(config=config)
            
            if brain.ctx is None:
                print("  [FAIL] GPU context not available")
                return False, {'error': 'No GPU context'}
            
            gpu_info = brain.ctx.info
            print(f"  GPU: {gpu_info.get('GL_RENDERER', 'Unknown')}")
            print(f"  Texture Size: {config.texture_size}×{config.texture_size}")
            
            # Calculate memory usage
            size = config.texture_size
            bytes_per_pixel = 16  # RGBA float32
            neural_mem = size * size * bytes_per_pixel
            connectivity_mem = size * size * bytes_per_pixel
            spatial_mem = size * size * bytes_per_pixel
            memory_mem = config.memory_texture_size * config.memory_texture_size * bytes_per_pixel
            total_mem_mb = (neural_mem + connectivity_mem + spatial_mem + memory_mem) / 1024 / 1024
            
            print(f"  Estimated Memory: {total_mem_mb:.1f} MB")
            
            # Try to run evolution
            print("  Testing evolution...")
            start_time = time.perf_counter()
            
            try:
                # Warmup
                brain.evolve(iterations=1)
                
                # Benchmark
                evolution_times = []
                for i in range(iterations):
                    step_start = time.perf_counter()
                    result = brain.evolve(iterations=1)
                    step_time = time.perf_counter() - step_start
                    evolution_times.append(step_time)
                    
                    if i == 0:
                        print(f"    First step: {step_time*1000:.2f}ms")
                
                avg_time = np.mean(evolution_times)
                min_time = np.min(evolution_times)
                max_time = np.max(evolution_times)
                
                neurons_per_second = neurons / avg_time
                gflops = (neurons * 25) / (avg_time * 1e9)  # 25 ops per neuron
                
                print(f"  ✓ SUCCESS")
                print(f"    Avg time per step: {avg_time*1000:.2f}ms")
                print(f"    Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
                print(f"    Throughput: {neurons_per_second/1e6:.2f}M neurons/s")
                print(f"    Compute: {gflops:.2f} GFLOPS")
                
                metrics = {
                    'neurons': neurons,
                    'texture_size': config.texture_size,
                    'memory_mb': total_mem_mb,
                    'success': True,
                    'avg_time_per_step': avg_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'neurons_per_second': neurons_per_second,
                    'gflops': gflops,
                    'gpu': gpu_info.get('GL_RENDERER', 'Unknown')
                }
                
                brain.release()
                return True, metrics
                
            except Exception as e:
                print(f"  ✗ FAILED during evolution: {e}")
                brain.release()
                return False, {'error': str(e), 'memory_mb': total_mem_mb}
                
        except MemoryError as e:
            print(f"  ✗ MEMORY ERROR: {e}")
            return False, {'error': 'MemoryError', 'message': str(e)}
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False, {'error': str(e)}
    
    def find_maximum_size(self, start_size: int = 1048576, max_attempts: int = 10):
        """
        Find maximum network size using binary search approach.
        
        Args:
            start_size: Starting network size to test
            max_attempts: Maximum number of size increments to try
        """
        print("\n" + "="*80)
        print("GPU STRESS TEST: Finding Maximum Network Size")
        print("="*80)
        print(f"Target GPU: RTX 3090 (24GB VRAM)")
        print(f"Starting size: {start_size:,} neurons")
        print("="*80)
        
        # Test progression: exponential growth then binary search
        test_sizes = []
        
        # Start with known good sizes
        base_sizes = [65536, 262144, 1048576]  # 256², 512², 1024²
        
        # Then try larger sizes
        current = start_size
        multiplier = 2
        
        for i in range(max_attempts):
            if current not in test_sizes:
                test_sizes.append(current)
            current = int(current * multiplier)
            if current > 500_000_000:  # Cap at 500M neurons (very large)
                break
        
        # Sort and test
        test_sizes = sorted(set(test_sizes))
        
        successful_tests = []
        failed_tests = []
        
        for size in test_sizes:
            success, metrics = self.test_network_size(size, iterations=3)
            
            if success:
                successful_tests.append(metrics)
                self.max_successful_size = max(self.max_successful_size, size)
                print(f"\n✓ {size:,} neurons: SUCCESS")
            else:
                failed_tests.append({'neurons': size, **metrics})
                if self.failure_point is None:
                    self.failure_point = size
                print(f"\n✗ {size:,} neurons: FAILED")
                
                # If we hit a failure, try sizes between last success and failure
                if successful_tests:
                    last_success = successful_tests[-1]['neurons']
                    # Try a few intermediate sizes
                    intermediate_sizes = [
                        int(last_success * 1.5),
                        int(last_success * 1.75),
                        int((last_success + size) / 2)
                    ]
                    
                    for inter_size in intermediate_sizes:
                        if inter_size > last_success and inter_size < size:
                            if inter_size not in [s['neurons'] for s in successful_tests]:
                                print(f"\n  Trying intermediate size: {inter_size:,} neurons...")
                                inter_success, inter_metrics = self.test_network_size(inter_size, iterations=2)
                                if inter_success:
                                    successful_tests.append(inter_metrics)
                                    self.max_successful_size = max(self.max_successful_size, inter_size)
                                else:
                                    failed_tests.append({'neurons': inter_size, **inter_metrics})
        
        self.results = {
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'max_successful_size': self.max_successful_size,
            'failure_point': self.failure_point,
            'gpu': successful_tests[0]['gpu'] if successful_tests else 'Unknown'
        }
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate professional report of stress test results."""
        report = []
        report.append("# GPU Stress Test Report: RTX 3090 Maximum Capacity\n\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if not self.results:
            report.append("No test results available.\n\n")
            return ''.join(report)
        
        successful = self.results.get('successful_tests', [])
        failed = self.results.get('failed_tests', [])
        
        report.append("## Executive Summary\n\n")
        report.append(f"This stress test determined the maximum network size that can be ")
        report.append(f"processed on an NVIDIA GeForce RTX 3090 GPU running NeuroCHIMERA.\n\n")
        
        if successful:
            max_test = max(successful, key=lambda x: x['neurons'])
            report.append(f"**Maximum Successful Network Size:** {max_test['neurons']:,} neurons\n\n")
            report.append(f"- **Texture Size:** {max_test['texture_size']}×{max_test['texture_size']}\n")
            report.append(f"- **Memory Usage:** {max_test['memory_mb']:.1f} MB\n")
            report.append(f"- **Performance:** {max_test['neurons_per_second']/1e6:.2f}M neurons/s\n")
            report.append(f"- **Compute:** {max_test['gflops']:.2f} GFLOPS\n\n")
        
        if self.failure_point:
            report.append(f"**Failure Point:** {self.failure_point:,} neurons\n\n")
        
        # Detailed results
        report.append("## Detailed Test Results\n\n")
        
        if successful:
            report.append("### Successful Tests\n\n")
            report.append("| Neurons | Texture | Memory (MB) | Time/Step (ms) | Throughput (M/s) | GFLOPS |\n")
            report.append("|---------|---------|-------------|----------------|------------------|--------|\n")
            
            for test in sorted(successful, key=lambda x: x['neurons']):
                report.append(f"| {test['neurons']:,} | {test['texture_size']}×{test['texture_size']} | ")
                report.append(f"{test['memory_mb']:.1f} | {test['avg_time_per_step']*1000:.2f} | ")
                report.append(f"{test['neurons_per_second']/1e6:.2f} | {test['gflops']:.2f} |\n")
            
            report.append("\n")
        
        if failed:
            report.append("### Failed Tests\n\n")
            report.append("| Neurons | Error |\n")
            report.append("|---------|-------|\n")
            
            for test in sorted(failed, key=lambda x: x.get('neurons', 0)):
                neurons = test.get('neurons', 'Unknown')
                error = test.get('error', 'Unknown error')
                report.append(f"| {neurons:,} | {error} |\n")
            
            report.append("\n")
        
        # Performance analysis
        if len(successful) > 1:
            report.append("## Performance Analysis\n\n")
            
            # Find peak performance
            peak_perf = max(successful, key=lambda x: x['neurons_per_second'])
            report.append(f"**Peak Throughput:** {peak_perf['neurons_per_second']/1e6:.2f}M neurons/s ")
            report.append(f"at {peak_perf['neurons']:,} neurons\n\n")
            
            # Memory efficiency
            report.append("### Memory Efficiency\n\n")
            report.append("Memory usage scales approximately linearly with network size.\n\n")
            
            # Performance scaling
            report.append("### Performance Scaling\n\n")
            report.append("Performance characteristics:\n\n")
            
            small_networks = [t for t in successful if t['neurons'] <= 262144]
            large_networks = [t for t in successful if t['neurons'] >= 1048576]
            
            if small_networks:
                avg_small = np.mean([t['neurons_per_second']/1e6 for t in small_networks])
                report.append(f"- **Small networks (≤262K):** Average {avg_small:.2f}M neurons/s\n")
            
            if large_networks:
                avg_large = np.mean([t['neurons_per_second']/1e6 for t in large_networks])
                report.append(f"- **Large networks (≥1M):** Average {avg_large:.2f}M neurons/s\n")
        
        # Conclusions
        report.append("## Conclusions\n\n")
        
        if successful:
            max_test = max(successful, key=lambda x: x['neurons'])
            report.append(f"1. **Maximum Viable Network:** {max_test['neurons']:,} neurons ")
            report.append(f"({max_test['texture_size']}×{max_test['texture_size']} texture)\n")
            report.append(f"2. **Memory Limit:** Approximately {max_test['memory_mb']:.1f} MB for maximum network\n")
            report.append(f"3. **Performance at Maximum:** {max_test['neurons_per_second']/1e6:.2f}M neurons/s\n")
            
            if len(successful) > 3:
                peak = max(successful, key=lambda x: x['neurons_per_second'])
                report.append(f"4. **Peak Performance:** {peak['neurons_per_second']/1e6:.2f}M neurons/s ")
                report.append(f"at {peak['neurons']:,} neurons\n")
        
        report.append("\n")
        
        return ''.join(report)
    
    def save_results(self, filename: str = "gpu_stress_test_results.json"):
        """Save results to JSON file."""
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
    
    def save_report(self, filename: str = "GPU_STRESS_TEST_REPORT.md"):
        """Save report to markdown file."""
        report = self.generate_report()
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        output_path = reports_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {output_path}")


def main():
    """Main stress test execution."""
    print("\n" + "="*80)
    print("GPU STRESS TEST: RTX 3090 Maximum Capacity")
    print("="*80)
    print("\nThis test will push the RTX 3090 to find the maximum network size")
    print("that can be processed. This may take several minutes.\n")
    
    stress_test = GPUStressTest()
    
    # Start from 1M neurons and go up
    results = stress_test.find_maximum_size(start_size=2097152, max_attempts=8)
    
    # Save results
    stress_test.save_results()
    stress_test.save_report()
    
    # Summary
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
    print("="*80)
    
    if results.get('successful_tests'):
        successful = results['successful_tests']
        max_test = max(successful, key=lambda x: x['neurons'])
        
        print(f"\n✓ Maximum Successful Size: {max_test['neurons']:,} neurons")
        print(f"  Texture: {max_test['texture_size']}×{max_test['texture_size']}")
        print(f"  Memory: {max_test['memory_mb']:.1f} MB")
        print(f"  Performance: {max_test['neurons_per_second']/1e6:.2f}M neurons/s")
        print(f"  Compute: {max_test['gflops']:.2f} GFLOPS")
        
        peak = max(successful, key=lambda x: x['neurons_per_second'])
        if peak['neurons'] != max_test['neurons']:
            print(f"\n  Peak Performance: {peak['neurons_per_second']/1e6:.2f}M neurons/s")
            print(f"    at {peak['neurons']:,} neurons")
    
    if results.get('failure_point'):
        print(f"\n✗ Failure Point: {results['failure_point']:,} neurons")
    
    print(f"\nTotal successful tests: {len(successful)}")
    print(f"Total failed tests: {len(results.get('failed_tests', []))}")
    print("\n[OK] Stress test completed")
    print("="*80)


if __name__ == '__main__':
    main()

