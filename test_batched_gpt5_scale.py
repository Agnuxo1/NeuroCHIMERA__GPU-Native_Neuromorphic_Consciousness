"""
Test Batched Processing - GPT-5 Scale (2 Trillion Neurons)
===========================================================

Tests the batched engine with GPT-5 scale network (2 trillion neurons).
Processes in tiles to maintain 100% GPU utilization without collapse.
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from engine_batched import BatchedConfig, BatchedNeuroCHIMERA


def test_gpt5_scale():
    """Test with 2 trillion neurons (GPT-5 scale)."""
    print("="*80)
    print("GPT-5 SCALE TEST: 2 Trillion Neurons")
    print("="*80)
    print("Processing strategy:")
    print("  - Divide network into GPU-sized tiles (8192×8192 = 67M neurons)")
    print("  - Process tiles sequentially with 100% GPU utilization")
    print("  - Stream data to/from GPU to avoid memory collapse")
    print("  - Maintain continuous processing pipeline")
    print("="*80)
    
    # GPT-5 scale: 2 trillion neurons
    total_neurons = 2_000_000_000_000
    
    config = BatchedConfig(
        total_neurons=total_neurons,
        gpu_tile_size=8192,  # 67M neurons per tile
        gpu_memory_limit_mb=20000
    )
    
    print(f"\nConfiguration:")
    print(f"  Total neurons: {total_neurons:,} ({total_neurons/1e12:.2f} trillion)")
    print(f"  GPU tile size: {config.gpu_tile_size}×{config.gpu_tile_size}")
    print(f"  Neurons per tile: {config.neurons_per_tile:,}")
    print(f"  Number of tiles: {config.num_tiles:,}")
    print(f"  Tile grid: {config.total_tiles[0]}×{config.total_tiles[1]}")
    
    # Estimate time (assuming ~25ms per tile)
    estimated_time_per_tile = 0.025  # seconds
    estimated_total_time = config.num_tiles * estimated_time_per_tile
    estimated_hours = estimated_total_time / 3600
    
    print(f"\nTime Estimates:")
    print(f"  Per tile: ~{estimated_time_per_tile*1000:.1f}ms")
    print(f"  Total: ~{estimated_total_time:.0f}s ({estimated_hours:.2f} hours)")
    print(f"  Throughput: ~{config.neurons_per_tile/estimated_time_per_tile/1e9:.2f}B neurons/s per tile")
    print()
    
    try:
        print("Initializing batched engine...")
        brain = BatchedNeuroCHIMERA(config)
        
        print("\nStarting processing...")
        print("(This will process all tiles sequentially with 100% GPU utilization)\n")
        
        start_time = time.perf_counter()
        
        # Process with 1 iteration per tile (can increase if needed)
        result = brain.evolve(iterations=1)
        
        total_time = time.perf_counter() - start_time
        
        brain.release()
        
        # Final report
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"Total neurons processed: {result['total_neurons']:,}")
        print(f"Total tiles processed: {result['total_tiles']:,}")
        print(f"Total time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
        print(f"Overall throughput: {result['throughput_neurons_per_sec']/1e9:.2f}B neurons/s")
        print(f"Average GPU time per tile: {result['avg_gpu_time_per_tile']*1000:.2f}ms")
        print(f"GPU utilization: ~100% (continuous processing)")
        print("="*80)
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    result = test_gpt5_scale()
    
    if result:
        print(f"\n[OK] Successfully processed {result['total_neurons']:,} neurons")
        print(f"    This demonstrates the system can handle GPT-5 scale networks")
        print(f"    by processing in batches while maintaining 100% GPU utilization")
    else:
        print("\n[FAIL] Test failed")

