"""
Quick GPU Utilization Test
===========================
Quick test to verify GPU utilization improvements.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import NeuroCHIMERAConfig
from engine_optimized import OptimizedNeuroCHIMERA
from engine_multi_core import MultiCoreNeuroCHIMERA
from Benchmarks.gpu_utilization_validator import GPUUtilizationValidator


def quick_test(neurons: int = 1_048_576):
    """Quick GPU utilization test."""
    print(f"\nQuick GPU Test - {neurons:,} neurons")
    print("=" * 60)
    
    # Test 1: Optimized Engine
    print("\n1. Testing Optimized Engine...")
    validator = GPUUtilizationValidator(target_min_utilization=60.0)
    validator.start_monitoring(interval=0.2)
    
    try:
        config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
        engine = OptimizedNeuroCHIMERA(config=config)
        
        if engine.ctx:
            # Run for 3 seconds
            start = time.time()
            while time.time() - start < 3.0:
                engine.evolve_optimized(iterations=1)
            
            engine.ctx.finish()
            engine.release()
        
    except Exception as e:
        print(f"  Error: {e}")
    
    analysis = validator.stop_monitoring()
    print(f"\n  GPU Utilization: {analysis['gpu_utilization']['mean']:.1f}% "
          f"± {analysis['gpu_utilization']['std']:.1f}%")
    print(f"  Verdict: {analysis['validation']['verdict']}")
    
    # Test 2: Multi-Core Engine
    print("\n2. Testing Multi-Core Engine...")
    validator2 = GPUUtilizationValidator(target_min_utilization=60.0)
    validator2.start_monitoring(interval=0.2)
    
    try:
        config = NeuroCHIMERAConfig(neurons=neurons, use_hns=True)
        engine = MultiCoreNeuroCHIMERA(config=config, parallel_batches=4)
        
        if engine.ctx:
            # Run for 3 seconds
            start = time.time()
            while time.time() - start < 3.0:
                engine.evolve_ultra_optimized(iterations=1)
            
            engine.ctx.finish()
            engine.release()
        
    except Exception as e:
        print(f"  Error: {e}")
    
    analysis2 = validator2.stop_monitoring()
    print(f"\n  GPU Utilization: {analysis2['gpu_utilization']['mean']:.1f}% "
          f"± {analysis2['gpu_utilization']['std']:.1f}%")
    print(f"  Verdict: {analysis2['validation']['verdict']}")
    
    print("\n" + "=" * 60)
    print("Quick test complete!")
    print("=" * 60)


if __name__ == '__main__':
    quick_test()
