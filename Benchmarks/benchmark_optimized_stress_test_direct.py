"""
Direct Optimized GPU Stress Test: Maximum Network Size
======================================================

Directly test specific texture sizes to find maximum capacity.
"""

import time
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine_optimized import OptimizedNeuroCHIMERA
from engine import NeuroCHIMERAConfig


def test_texture_size(tex_size: int, iterations: int = 3) -> Tuple[bool, Dict]:
    """Test a specific texture size."""
    neurons = tex_size * tex_size
    print(f"\n{'='*80}")
    print(f"Testing {tex_size}×{tex_size} texture ({neurons:,} neurons)")
    print(f"{'='*80}")
    
    try:
        config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
        brain = OptimizedNeuroCHIMERA(config=config)
        
        if brain.ctx is None:
            print("  [FAIL] GPU context not available")
            return False, {'error': 'No GPU context'}
        
        gpu_info = brain.ctx.info
        print(f"  GPU: {gpu_info.get('GL_RENDERER', 'Unknown')}")
        
        # Calculate memory
        size = config.texture_size
        bytes_per_pixel = 16
        neural_mem = size * size * bytes_per_pixel
        connectivity_mem = size * size * bytes_per_pixel
        spatial_mem = size * size * bytes_per_pixel
        memory_mem = config.memory_texture_size * config.memory_texture_size * bytes_per_pixel
        total_mem_mb = (neural_mem + connectivity_mem + spatial_mem + memory_mem) / 1024 / 1024
        
        print(f"  Memory: {total_mem_mb:.1f} MB")
        
        # Test evolution
        print("  Testing evolution...")
        evolution_times = []
        
        for i in range(iterations):
            step_start = time.perf_counter()
            result = brain.evolve_optimized(iterations=1)
            brain.ctx.finish()
            step_time = time.perf_counter() - step_start
            evolution_times.append(step_time)
            
            if i == 0:
                print(f"    First step: {step_time*1000:.2f}ms")
        
        avg_time = np.mean(evolution_times)
        neurons_per_second = config.neurons / avg_time
        gflops = (config.neurons * 25) / (avg_time * 1e9)
        
        print(f"  [OK] SUCCESS")
        print(f"    Avg: {avg_time*1000:.2f}ms, Throughput: {neurons_per_second/1e6:.2f}M neurons/s")
        print(f"    Compute: {gflops:.2f} GFLOPS")
        
        metrics = {
            'neurons': config.neurons,
            'texture_size': config.texture_size,
            'memory_mb': total_mem_mb,
            'avg_time_per_step': avg_time,
            'neurons_per_second': neurons_per_second,
            'gflops': gflops,
            'gpu': gpu_info.get('GL_RENDERER', 'Unknown')
        }
        
        brain.release()
        return True, metrics
        
    except MemoryError as e:
        print(f"  [FAIL] MEMORY ERROR: {e}")
        return False, {'error': 'MemoryError', 'message': str(e)}
    except Exception as e:
        print(f"  [FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


def main():
    """Test specific texture sizes to find maximum."""
    print("\n" + "="*80)
    print("DIRECT OPTIMIZED GPU STRESS TEST")
    print("="*80)
    print("Testing specific texture sizes to find maximum capacity")
    print("="*80)
    
    # Test sizes in order
    texture_sizes = [
        4096,   # 16M - known good
        6144,   # 37M
        8192,   # 67M - known good
        10240,  # 104M
        12288,  # 150M
        14336,  # 205M
        16384,  # 268M
        18432,  # 339M
        20480,  # 419M
    ]
    
    successful = []
    failed = []
    
    for tex_size in texture_sizes:
        success, metrics = test_texture_size(tex_size, iterations=3)
        
        if success:
            successful.append(metrics)
            print(f"\n[OK] {tex_size}×{tex_size} ({metrics['neurons']:,} neurons): SUCCESS")
        else:
            failed.append({'texture_size': tex_size, **metrics})
            print(f"\n[FAIL] {tex_size}×{tex_size}: FAILED")
            
            # If we hit a failure, stop testing larger sizes
            if failed:
                print(f"\nStopping at failure point: {tex_size}×{tex_size}")
                break
    
    # Save results
    results = {
        'successful_tests': successful,
        'failed_tests': failed,
        'max_successful_size': max([s['neurons'] for s in successful]) if successful else 0,
        'gpu': successful[0]['gpu'] if successful else 'Unknown',
        'implementation': 'Optimized (Compute Shaders)'
    }
    
    # Save JSON
    output_path = Path(__file__).parent / "optimized_gpu_stress_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    # Generate report
    report = []
    report.append("# Optimized GPU Stress Test Report: RTX 3090 Maximum Capacity\n\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append("## Executive Summary\n\n")
    
    if successful:
        max_test = max(successful, key=lambda x: x['neurons'])
        report.append(f"**Maximum Successful Network Size:** {max_test['neurons']:,} neurons\n\n")
        report.append(f"- **Texture Size:** {max_test['texture_size']}×{max_test['texture_size']}\n")
        report.append(f"- **Memory Usage:** {max_test['memory_mb']:.1f} MB\n")
        report.append(f"- **Performance:** {max_test['neurons_per_second']/1e6:.2f}M neurons/s\n")
        report.append(f"- **Compute:** {max_test['gflops']:.2f} GFLOPS\n")
        report.append(f"- **Time per Step:** {max_test['avg_time_per_step']*1000:.2f}ms\n\n")
    
    report.append("## Detailed Results\n\n")
    report.append("| Texture | Neurons | Memory (MB) | Time/Step (ms) | Throughput (M/s) | GFLOPS |\n")
    report.append("|---------|---------|-------------|----------------|------------------|--------|\n")
    
    for test in sorted(successful, key=lambda x: x['texture_size']):
        report.append(f"| {test['texture_size']}×{test['texture_size']} | {test['neurons']:,} | ")
        report.append(f"{test['memory_mb']:.1f} | {test['avg_time_per_step']*1000:.2f} | ")
        report.append(f"{test['neurons_per_second']/1e6:.2f} | {test['gflops']:.2f} |\n")
    
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "OPTIMIZED_GPU_STRESS_TEST_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    print(f"Report saved to {report_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if successful:
        max_test = max(successful, key=lambda x: x['neurons'])
        print(f"\n[OK] Maximum Successful: {max_test['neurons']:,} neurons")
        print(f"     Texture: {max_test['texture_size']}×{max_test['texture_size']}")
        print(f"     Memory: {max_test['memory_mb']:.1f} MB")
        print(f"     Performance: {max_test['neurons_per_second']/1e6:.2f}M neurons/s")
        print(f"     Compute: {max_test['gflops']:.2f} GFLOPS")
    
    if failed:
        print(f"\n[FAIL] First Failure: {failed[0].get('texture_size', 'Unknown')}×{failed[0].get('texture_size', 'Unknown')}")
    
    print(f"\nTotal successful: {len(successful)}")
    print(f"Total failed: {len(failed)}")
    print("="*80)


if __name__ == '__main__':
    main()

