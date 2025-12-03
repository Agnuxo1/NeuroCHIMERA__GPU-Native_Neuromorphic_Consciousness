"""
GPU Utilization Validator
==========================
Validates that CHIMERA benchmarks execute 100% on GPU with proper utilization.

Features:
- Real-time GPU monitoring during benchmarks
- Detection of CPU fallbacks
- Validation of sustained 80-100% GPU utilization
- Statistical analysis of GPU usage patterns
"""

import subprocess
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class GPUUtilizationValidator:
    """Validates GPU execution and monitors utilization."""
    
    def __init__(self, target_min_utilization: float = 80.0):
        self.target_min_utilization = target_min_utilization
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
        
    def get_gpu_metrics(self) -> Optional[Dict]:
        """Query current GPU metrics."""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'gpu_util_percent': float(values[0]),
                    'mem_util_percent': float(values[1]),
                    'mem_used_mb': float(values[2]),
                    'mem_total_mb': float(values[3]),
                    'temp_celsius': float(values[4]),
                    'power_draw_watts': float(values[5]),
                    'gpu_clock_mhz': float(values[6]),
                    'mem_clock_mhz': float(values[7]),
                    'timestamp': time.time()
                }
        except Exception as e:
            print(f"Warning: GPU metrics unavailable: {e}")
        return None
    
    def _monitor_loop(self, interval: float = 0.1):
        """Background monitoring loop."""
        while self.monitoring:
            metrics = self.get_gpu_metrics()
            if metrics:
                self.samples.append(metrics)
            time.sleep(interval)
    
    def start_monitoring(self, interval: float = 0.1):
        """Start background GPU monitoring."""
        self.monitoring = True
        self.samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"[GPU Monitor] Started (sampling every {interval}s)")
    
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return analysis."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if not self.samples:
            return {'error': 'No samples collected'}
        
        # Analyze samples
        gpu_utils = [s['gpu_util_percent'] for s in self.samples]
        mem_utils = [s['mem_util_percent'] for s in self.samples]
        
        analysis = {
            'sample_count': len(self.samples),
            'duration_seconds': self.samples[-1]['timestamp'] - self.samples[0]['timestamp'],
            'gpu_utilization': {
                'mean': np.mean(gpu_utils),
                'std': np.std(gpu_utils),
                'min': np.min(gpu_utils),
                'max': np.max(gpu_utils),
                'median': np.median(gpu_utils),
                'percentile_95': np.percentile(gpu_utils, 95),
                'percentile_5': np.percentile(gpu_utils, 5),
            },
            'memory_utilization': {
                'mean': np.mean(mem_utils),
                'std': np.std(mem_utils),
                'min': np.min(mem_utils),
                'max': np.max(mem_utils),
            },
            'validation': self._validate_gpu_usage(gpu_utils),
            'samples': self.samples
        }
        
        print(f"\n[GPU Monitor] Stopped - {len(self.samples)} samples collected")
        print(f"  GPU Utilization: {analysis['gpu_utilization']['mean']:.1f}% ± {analysis['gpu_utilization']['std']:.1f}%")
        print(f"  Range: {analysis['gpu_utilization']['min']:.1f}% - {analysis['gpu_utilization']['max']:.1f}%")
        
        return analysis
    
    def _validate_gpu_usage(self, gpu_utils: List[float]) -> Dict:
        """Validate GPU usage patterns."""
        mean_util = np.mean(gpu_utils)
        std_util = np.std(gpu_utils)
        
        # Check for CPU fallback (consistently low GPU usage)
        cpu_fallback = mean_util < 20.0
        
        # Check for optimal usage
        optimal = mean_util >= self.target_min_utilization
        
        # Check for erratic behavior (high variance)
        erratic = std_util > 30.0
        
        # Check for spikes (max >> mean)
        max_util = np.max(gpu_utils)
        spiky = (max_util > 95) and (mean_util < 50)
        
        validation = {
            'passed': optimal and not cpu_fallback and not erratic,
            'cpu_fallback_detected': cpu_fallback,
            'optimal_utilization': optimal,
            'erratic_behavior': erratic,
            'spike_pattern_detected': spiky,
            'mean_utilization': mean_util,
            'target_utilization': self.target_min_utilization,
        }
        
        # Generate verdict
        if cpu_fallback:
            validation['verdict'] = 'FAIL - CPU fallback detected (low GPU usage)'
        elif spiky:
            validation['verdict'] = 'FAIL - Spiky GPU usage (not sustained)'
        elif erratic:
            validation['verdict'] = 'WARN - Erratic GPU usage (high variance)'
        elif optimal:
            validation['verdict'] = 'PASS - Optimal GPU utilization'
        else:
            validation['verdict'] = f'WARN - Below target ({mean_util:.1f}% < {self.target_min_utilization}%)'
        
        return validation
    
    def save_results(self, filename: str, analysis: Dict):
        """Save monitoring results to JSON."""
        output_path = Path(filename)
        
        # Save full results
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"[GPU Monitor] Results saved to: {output_path}")
        
        # Also save summary
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("GPU Utilization Validation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Samples: {analysis['sample_count']}\n")
            f.write(f"Duration: {analysis['duration_seconds']:.2f}s\n\n")
            
            f.write("GPU Utilization:\n")
            util = analysis['gpu_utilization']
            f.write(f"  Mean:   {util['mean']:.1f}%\n")
            f.write(f"  Std:    {util['std']:.1f}%\n")
            f.write(f"  Min:    {util['min']:.1f}%\n")
            f.write(f"  Max:    {util['max']:.1f}%\n")
            f.write(f"  Median: {util['median']:.1f}%\n\n")
            
            f.write("Validation:\n")
            val = analysis['validation']
            f.write(f"  Verdict: {val['verdict']}\n")
            f.write(f"  CPU Fallback: {'YES' if val['cpu_fallback_detected'] else 'NO'}\n")
            f.write(f"  Optimal: {'YES' if val['optimal_utilization'] else 'NO'}\n")
            f.write(f"  Erratic: {'YES' if val['erratic_behavior'] else 'NO'}\n")
            f.write(f"  Spiky: {'YES' if val['spike_pattern_detected'] else 'NO'}\n")
        
        print(f"[GPU Monitor] Summary saved to: {summary_path}")


def validate_benchmark(benchmark_func, benchmark_name: str, 
                       target_min_util: float = 80.0, **kwargs) -> Dict:
    """
    Run a benchmark with GPU validation.
    
    Args:
        benchmark_func: Function to benchmark
        benchmark_name: Name for logging
        target_min_util: Minimum target GPU utilization
        **kwargs: Arguments to pass to benchmark_func
    
    Returns:
        Dict with benchmark results and GPU validation
    """
    print(f"\n{'='*80}")
    print(f"Running: {benchmark_name}")
    print(f"{'='*80}\n")
    
    validator = GPUUtilizationValidator(target_min_utilization=target_min_util)
    
    # Start monitoring
    validator.start_monitoring(interval=0.1)
    
    # Small delay to ensure monitoring starts
    time.sleep(0.5)
    
    # Run benchmark
    start_time = time.time()
    try:
        benchmark_result = benchmark_func(**kwargs)
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        benchmark_result = {'error': str(e)}
    
    elapsed = time.time() - start_time
    
    # Stop monitoring
    gpu_analysis = validator.stop_monitoring()
    
    # Combine results
    result = {
        'benchmark_name': benchmark_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'benchmark_result': benchmark_result,
        'gpu_validation': gpu_analysis,
    }
    
    # Print validation verdict
    print(f"\n{'='*80}")
    print(f"GPU Validation: {gpu_analysis['validation']['verdict']}")
    print(f"{'='*80}\n")
    
    return result


if __name__ == '__main__':
    print("GPU Utilization Validator")
    print("=" * 60)
    print("This module provides GPU monitoring and validation.")
    print("Import and use validate_benchmark() to wrap your benchmarks.")
    print("=" * 60)
    
    # Quick test
    validator = GPUUtilizationValidator()
    metrics = validator.get_gpu_metrics()
    if metrics:
        print("\nCurrent GPU Status:")
        print(f"  GPU Utilization: {metrics['gpu_util_percent']}%")
        print(f"  Memory: {metrics['mem_used_mb']:.0f} / {metrics['mem_total_mb']:.0f} MB")
        print(f"  Temperature: {metrics['temp_celsius']}°C")
        print(f"  Power: {metrics['power_draw_watts']}W")
    else:
        print("\n[WARNING] GPU metrics unavailable - ensure nvidia-smi is accessible")
