"""
Comprehensive HNS Performance Benchmarks
========================================

Enhanced HNS benchmarks covering:
- Precision tests (CPU & GPU)
- Speed benchmarks
- Accumulation tests
- GPU acceleration comparison
"""

import time
import numpy as np
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_number import (
    HNumber, hns_add, hns_scale, hns_normalize,
    hns_multiply, BASE, hns_add_batch, hns_scale_batch, hns_normalize_batch
)

try:
    import moderngl
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Warning: moderngl not available. GPU benchmarks will be skipped.")


class HNSBenchmark:
    """Comprehensive HNS benchmarking suite."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_precision_large_numbers(self) -> Dict:
        """Test precision with large numbers where float fails."""
        print("\n" + "="*80)
        print("PRECISION TEST: Large Numbers")
        print("="*80)
        
        results = {'test_name': 'Precision with large numbers', 'cases': []}
        
        # Test cases within HNS range (up to ~10^12)
        test_cases = [
            ("999,999 + 1", 999999.0, 1.0, 1000000.0),
            ("1,000,000,000 + 1", 1000000000.0, 1.0, 1000000001.0),
            ("999,999,999,999 + 1", 999999999999.0, 1.0, 1000000000000.0),
            ("500,500,500 + 600,600,600", 500500500.0, 600600600.0, 1101101100.0),
            ("123,456,789 + 987,654,321", 123456789.0, 987654321.0, 1111111110.0),
        ]
        
        for name, a_val, b_val, expected in test_cases:
            # Standard float
            float_result = a_val + b_val
            float_error = abs(float_result - expected)
            
            # HNS
            hns_a = HNumber(a_val)
            hns_b = HNumber(b_val)
            hns_result = hns_add(hns_a, hns_b)
            hns_error = abs(hns_result.to_float() - expected)
            
            case_result = {
                'name': name,
                'expected': expected,
                'float_result': float_result,
                'float_error': float_error,
                'hns_result': hns_result.to_float(),
                'hns_error': hns_error,
                'hns_wins': hns_error < float_error
            }
            
            results['cases'].append(case_result)
            
            print(f"\n{name}:")
            print(f"  Expected: {expected}")
            print(f"  Float: {float_result} (Error: {float_error:.2e})")
            print(f"  HNS:   {hns_result.to_float()} (Error: {hns_error:.2e})")
            if hns_error < float_error:
                improvement = ((float_error - hns_error) / float_error * 100) if float_error > 0 else 0
                print(f"  [OK] HNS is {improvement:.1f}% more precise")
        
        return results
    
    def benchmark_accumulative_precision(self, iterations: int = 1000000) -> Dict:
        """Test accumulative precision loss."""
        print("\n" + "="*80)
        print(f"ACCUMULATIVE PRECISION TEST ({iterations:,} iterations)")
        print("="*80)
        
        increment = 0.000001
        expected = iterations * increment
        
        # Standard float
        float_value = 0.0
        start_time = time.perf_counter()
        for _ in range(iterations):
            float_value += increment
        float_time = time.perf_counter() - start_time
        float_error = abs(float_value - expected)
        
        # HNS
        hns_value = HNumber(0.0)
        hns_increment = HNumber(increment)
        start_time = time.perf_counter()
        for _ in range(iterations):
            hns_value = hns_add(hns_value, hns_increment)
        hns_time = time.perf_counter() - start_time
        hns_error = abs(hns_value.to_float() - expected)
        
        print(f"\nExpected value: {expected}")
        print(f"\nStandard float:")
        print(f"  Result: {float_value:.10f}")
        print(f"  Error:  {float_error:.2e} ({float_error/expected*100:.6f}% relative)")
        print(f"  Time:   {float_time:.4f}s ({iterations/float_time:,.0f} ops/s)")
        print(f"\nHNS:")
        print(f"  Result: {hns_value.to_float():.10f}")
        print(f"  Error:  {hns_error:.2e} ({hns_error/expected*100:.6f}% relative)")
        print(f"  Time:   {hns_time:.4f}s ({iterations/hns_time:,.0f} ops/s)")
        print(f"  Overhead: {hns_time/float_time:.2f}x slower")
        
        return {
            'iterations': iterations,
            'expected': expected,
            'float': {'result': float_value, 'error': float_error, 'time': float_time},
            'hns': {'result': hns_value.to_float(), 'error': hns_error, 'time': hns_time},
            'hns_better': hns_error < float_error
        }
    
    def benchmark_speed_operations(self, iterations: int = 100000) -> Dict:
        """Test operation speed."""
        print("\n" + "="*80)
        print(f"SPEED BENCHMARK ({iterations:,} iterations)")
        print("="*80)
        
        results = {}
        
        # Prepare test data
        float_a = 123456.789
        float_b = 987654.321
        hns_a = HNumber(float_a)
        hns_b = HNumber(float_b)
        
        # ADDITION
        print("\n--- ADDITION ---")
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = float_a + float_b
        float_add_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = hns_add(hns_a, hns_b)
        hns_add_time = time.perf_counter() - start
        
        print(f"Float: {float_add_time*1000:.4f}ms ({iterations/float_add_time:,.0f} ops/s)")
        print(f"HNS:   {hns_add_time*1000:.4f}ms ({iterations/hns_add_time:,.0f} ops/s) - {hns_add_time/float_add_time:.2f}x slower")
        
        results['add'] = {
            'float': float_add_time,
            'hns': hns_add_time,
            'overhead': hns_add_time / float_add_time
        }
        
        # SCALING
        print("\n--- SCALAR MULTIPLICATION ---")
        scalar = 2.5
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = float_a * scalar
        float_scale_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = hns_scale(hns_a, scalar)
        hns_scale_time = time.perf_counter() - start
        
        print(f"Float: {float_scale_time*1000:.4f}ms ({iterations/float_scale_time:,.0f} ops/s)")
        print(f"HNS:   {hns_scale_time*1000:.4f}ms ({iterations/hns_scale_time:,.0f} ops/s) - {hns_scale_time/float_scale_time:.2f}x slower")
        
        results['scale'] = {
            'float': float_scale_time,
            'hns': hns_scale_time,
            'overhead': hns_scale_time / float_scale_time
        }
        
        return results
    
    def benchmark_batch_operations(self, batch_size: int = 10000) -> Dict:
        """Test batch operations performance."""
        print("\n" + "="*80)
        print(f"BATCH OPERATIONS BENCHMARK ({batch_size:,} elements)")
        print("="*80)
        
        # Generate random data
        hns_data_a = np.random.uniform(0, 1000, (batch_size, 4)).astype(np.float32)
        hns_data_b = np.random.uniform(0, 1000, (batch_size, 4)).astype(np.float32)
        scalars = np.random.uniform(0, 1, batch_size).astype(np.float32)
        
        # Normalize batch
        start = time.perf_counter()
        normalized = hns_normalize_batch(hns_data_a)
        normalize_time = time.perf_counter() - start
        
        # Add batch
        start = time.perf_counter()
        added = hns_add_batch(normalized, hns_data_b)
        add_time = time.perf_counter() - start
        
        # Scale batch
        start = time.perf_counter()
        scaled = hns_scale_batch(added, scalars)
        scale_time = time.perf_counter() - start
        
        total_time = normalize_time + add_time + scale_time
        
        print(f"\nNormalize: {normalize_time*1000:.4f}ms ({batch_size/normalize_time/1e6:.2f}M ops/s)")
        print(f"Add:      {add_time*1000:.4f}ms ({batch_size/add_time/1e6:.2f}M ops/s)")
        print(f"Scale:    {scale_time*1000:.4f}ms ({batch_size/scale_time/1e6:.2f}M ops/s)")
        print(f"Total:    {total_time*1000:.4f}ms ({batch_size/total_time/1e6:.2f}M ops/s)")
        
        return {
            'batch_size': batch_size,
            'normalize_time': normalize_time,
            'add_time': add_time,
            'scale_time': scale_time,
            'total_time': total_time,
            'throughput': batch_size / total_time
        }
    
    def run_all(self) -> Dict:
        """Run all benchmarks."""
        print("\n" + "="*80)
        print("COMPREHENSIVE HNS BENCHMARK SUITE")
        print("="*80)
        print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = {}
        
        # Precision tests
        all_results['precision'] = self.benchmark_precision_large_numbers()
        
        # Accumulative precision
        all_results['accumulative'] = self.benchmark_accumulative_precision(iterations=1000000)
        
        # Speed benchmarks
        all_results['speed'] = self.benchmark_speed_operations(iterations=100000)
        
        # Batch operations
        all_results['batch'] = self.benchmark_batch_operations(batch_size=10000)
        
        self.results = all_results
        return all_results
    
    def save_results(self, filename: str = "hns_benchmark_results.json"):
        """Save results to JSON file."""
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


def main():
    """Main benchmark execution."""
    benchmark = HNSBenchmark()
    results = benchmark.run_all()
    benchmark.save_results()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if 'precision' in results:
        wins = sum(1 for c in results['precision']['cases'] if c.get('hns_wins', False))
        total = len(results['precision']['cases'])
        print(f"\nPrecision: HNS better in {wins}/{total} cases ({wins/total*100:.1f}%)")
    
    if 'accumulative' in results:
        acc = results['accumulative']
        print(f"\nAccumulative Precision:")
        print(f"  Float error: {acc['float']['error']:.2e}")
        print(f"  HNS error:   {acc['hns']['error']:.2e}")
        print(f"  HNS better:  {acc['hns_better']}")
    
    if 'speed' in results:
        avg_overhead = (results['speed']['add']['overhead'] + results['speed']['scale']['overhead']) / 2
        print(f"\nSpeed: Average overhead {avg_overhead:.2f}x")
    
    print("\n[OK] Benchmark completed")


if __name__ == '__main__':
    main()

