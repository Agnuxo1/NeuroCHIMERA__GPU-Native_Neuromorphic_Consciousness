"""
FIXED: HNS Pure Benchmark with Validation
==========================================
Corrects methodological issues from previous version.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:
    HAS_MODERNGL = False


HNS_ADD_SHADER = """
#version 430

layout(local_size_x = 32, local_size_y = 32) in;

layout(std430, binding = 0) buffer BufA { vec4 data_a[]; };
layout(std430, binding = 1) buffer BufB { vec4 data_b[]; };
layout(std430, binding = 2) buffer BufOut { vec4 data_out[]; };

const float BASE = 1000.0;

void main() {
    uint idx = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    
    if (idx >= data_a.length()) return;
    
    vec4 a = data_a[idx];
    vec4 b = data_b[idx];
    
    // HNS Addition with carry propagation
    float sum0 = a.r + b.r;
    float carry0 = floor(sum0 / BASE);
    data_out[idx].r = mod(sum0, BASE);
    
    float sum1 = a.g + b.g + carry0;
    float carry1 = floor(sum1 / BASE);
    data_out[idx].g = mod(sum1, BASE);
    
    float sum2 = a.b + b.b + carry1;
    float carry2 = floor(sum2 / BASE);
    data_out[idx].b = mod(sum2, BASE);
    
    float sum3 = a.a + b.a + carry2;
    data_out[idx].a = mod(sum3, BASE);
}
"""


def validate_hns_add_cpu(a: np.ndarray, b: np.ndarray, base: float = 1000.0) -> np.ndarray:
    """CPU reference implementation for validation."""
    result = np.zeros_like(a)
    carry = 0.0
    
    for level in range(4):  # R, G, B, A
        sum_val = a[:, level] + b[:, level] + carry
        result[:, level] = np.mod(sum_val, base)
        carry = np.floor(sum_val / base)
    
    return result


def benchmark_hns_add_validated(ctx, size: int, runs: int = 20) -> dict:
    """Benchmark WITH validation of correctness."""
    print(f"\n  Testing HNS Addition - {size:,} elements (VALIDATED)")
    
    # Compile shader
    shader = ctx.compute_shader(HNS_ADD_SHADER)
    
    # Create test data
    rng = np.random.RandomState(42)
    data_a = rng.randint(0, 1000, size=(size, 4)).astype(np.float32)
    data_b = rng.randint(0, 1000, size=(size, 4)).astype(np.float32)
    
    buf_a = ctx.buffer(data_a.tobytes())
    buf_b = ctx.buffer(data_b.tobytes())
    buf_out = ctx.buffer(reserve=size * 16)
    
    buf_a.bind_to_storage_buffer(0)
    buf_b.bind_to_storage_buffer(1)
    buf_out.bind_to_storage_buffer(2)
    
    work_groups = (size + 1023) // 1024
    
    # Extended warmup (10 iterations)
    print(f"    Warming up GPU...")
    for _ in range(10):
        shader.run(work_groups, 1, 1)
    ctx.finish()
    
    # Run once for validation
    shader.run(work_groups, 1, 1)
    ctx.finish()
    
    # CRITICAL: Validate correctness
    print(f"    Validating correctness...")
    gpu_result = np.frombuffer(buf_out.read(), dtype=np.float32).reshape(size, 4)
    cpu_expected = validate_hns_add_cpu(data_a, data_b)
    
    max_error = np.max(np.abs(gpu_result - cpu_expected))
    if max_error > 0.01:
        raise ValueError(f"GPU shader produced incorrect results! Max error: {max_error}")
    print(f"    ✓ Validation passed (max error: {max_error:.2e})")
    
    # Now benchmark (measuring shader execution only, not sync)
    print(f"    Benchmarking {runs} runs...")
    
    # Method 1: Include ctx.finish() in timing (matches likely paper methodology)
    times_with_sync = []
    for i in range(runs):
        start = time.perf_counter()
        shader.run(work_groups, 1, 1)
        ctx.finish()
        elapsed = time.perf_counter() - start
        times_with_sync.append(elapsed)
    
    # Method 2: Batch without sync (pure GPU time)
    batch_size = 100
    start = time.perf_counter()
    for i in range(batch_size):
        shader.run(work_groups, 1, 1)
    ctx.finish()
    elapsed_batch = time.perf_counter() - start
    time_per_op_nosync = elapsed_batch / batch_size
    
    # Release
    buf_a.release()
    buf_b.release()
    buf_out.release()
    shader.release()
    
    mean_with_sync = np.mean(times_with_sync)
    std_with_sync = np.std(times_with_sync)
    
    # Throughput calculations
    # CLARIFICATION: 1 "operation" = processing 1 HNS element (vec4)
    # This involves ~8 arithmetic ops but counts as 1 HNS operation
    throughput_with_sync = size / mean_with_sync
    throughput_nosync = size / time_per_op_nosync
    
    print(f"    With sync: {mean_with_sync*1000:.3f}ms ± {std_with_sync*1000:.3f}ms")
    print(f"    → Throughput: {throughput_with_sync/1e9:.3f} billion ops/s")
    print(f"    Without sync: {time_per_op_nosync*1000:.3f}ms")
    print(f"    → Throughput: {throughput_nosync/1e9:.3f} billion ops/s")
    
    return {
        'operation': 'addition',
        'size': size,
        'runs': runs,
        'validated': True,
        'max_validation_error': float(max_error),
        'with_sync': {
            'mean_time_ms': mean_with_sync * 1000,
            'std_time_ms': std_with_sync * 1000,
            'throughput_ops_per_sec': throughput_with_sync,
            'throughput_billion_ops_per_sec': throughput_with_sync / 1e9
        },
        'without_sync': {
            'mean_time_ms': time_per_op_nosync * 1000,
            'throughput_ops_per_sec': throughput_nosync,
            'throughput_billion_ops_per_sec': throughput_nosync / 1e9
        },
        'note': '1 operation = 1 HNS element processed (vec4 with carry propagation)'
    }


def main():
    """Run VALIDATED HNS benchmark."""
    print("\n" + "="*80)
    print("VALIDATED HNS BENCHMARK - Scientific Rigor")
    print("="*80)
    
    if not HAS_MODERNGL:
        print("[ERROR] ModernGL required")
        return
    
    try:
        ctx = moderngl.create_standalone_context(require=430)
        print(f"\nGPU: {ctx.info['GL_RENDERER']}")
        print(f"OpenGL: {ctx.info['GL_VERSION']}")
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    test_sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    
    results = {
        'benchmark': 'Validated HNS Operations',
        'timestamp': datetime.now().isoformat(),
        'gpu': ctx.info['GL_RENDERER'],
        'methodology': {
            'warmup_iterations': 10,
            'measurement_runs': 20,
            'validation': 'CPU reference implementation',
            'timing': 'Both with and without ctx.finish() sync'
        },
        'results': []
    }
    
    print("\n" + "="*80)
    print("HNS ADDITION - VALIDATED BENCHMARK")
    print("="*80)
    
    for size in test_sizes:
        try:
            result = benchmark_hns_add_validated(ctx, size, runs=20)
            results['results'].append(result)
        except Exception as e:
            print(f"  [ERROR] Size {size}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save
    output_file = Path(__file__).parent / 'validated_hns_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Honest summary
    print("\n" + "="*80)
    print("HONEST ASSESSMENT")
    print("="*80)
    print("\nThis benchmark:")
    print("✓ Validates correctness against CPU reference")
    print("✓ Extensive warmup (10 iterations)")
    print("✓ Reports both with/without sync")
    print("✓ Clarifies 1 operation = 1 HNS element (vec4)")
    print("\nLimitations:")
    print("- Cannot confirm paper used identical methodology")
    print("- Sync overhead may vary by driver/system")
    print("- Results depend on thermal/power state")
    print("="*80)
    
    ctx.release()


if __name__ == '__main__':
    main()
