"""
Pure HNS Operations Benchmark
==============================
Validates paper claim of 19.8 billion HNS ops/s.

Tests ONLY HNS add and scale operations at various sizes,
matching the methodology described in the paper.
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
    print("[ERROR] ModernGL required")
    sys.exit(1)


# Pure HNS Addition Shader (matches paper implementation)
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
    
    // Level 0 (R): Units
    float sum0 = a.r + b.r;
    float carry0 = floor(sum0 / BASE);
    data_out[idx].r = mod(sum0, BASE);
    
    // Level 1 (G): Thousands
    float sum1 = a.g + b.g + carry0;
    float carry1 = floor(sum1 / BASE);
    data_out[idx].g = mod(sum1, BASE);
    
    // Level 2 (B): Millions
    float sum2 = a.b + b.b + carry1;
    float carry2 = floor(sum2 / BASE);
    data_out[idx].b = mod(sum2, BASE);
    
    // Level 3 (A): Billions
    float sum3 = a.a + b.a + carry2;
    data_out[idx].a = mod(sum3, BASE);
}
"""

# Pure HNS Scaling Shader
HNS_SCALE_SHADER = """
#version 430

layout(local_size_x = 32, local_size_y = 32) in;

layout(std430, binding = 0) buffer BufA { vec4 data_a[]; };
layout(std430, binding = 1) buffer BufOut { vec4 data_out[]; };

uniform float u_scale;

const float BASE = 1000.0;

void main() {
    uint idx = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    
    if (idx >= data_a.length()) return;
    
    vec4 a = data_a[idx];
    
    // Scale all levels
    float s0 = a.r * u_scale;
    float carry0 = floor(s0 / BASE);
    data_out[idx].r = mod(s0, BASE);
    
    float s1 = a.g * u_scale + carry0;
    float carry1 = floor(s1 / BASE);
    data_out[idx].g = mod(s1, BASE);
    
    float s2 = a.b * u_scale + carry1;
    float carry2 = floor(s2 / BASE);
    data_out[idx].b = mod(s2, BASE);
    
    float s3 = a.a * u_scale + carry2;
    data_out[idx].a = mod(s3, BASE);
}
"""


def benchmark_hns_add(ctx, size: int, runs: int = 20) -> dict:
    """Benchmark pure HNS addition operations."""
    print(f"\n  Testing HNS Addition - {size:,} elements")
    
    # Compile shader
    shader = ctx.compute_shader(HNS_ADD_SHADER)
    
    # Create buffers
    rng = np.random.RandomState(42)
    data_a = rng.randint(0, 1000, size=(size, 4)).astype(np.float32)
    data_b = rng.randint(0, 1000, size=(size, 4)).astype(np.float32)
    
    buf_a = ctx.buffer(data_a.tobytes())
    buf_b = ctx.buffer(data_b.tobytes())
    buf_out = ctx.buffer(reserve=size * 16)  # 4 floats * 4 bytes
    
    # Bind buffers
    buf_a.bind_to_storage_buffer(0)
    buf_b.bind_to_storage_buffer(1)
    buf_out.bind_to_storage_buffer(2)
    
    # Calculate work groups
    work_groups = (size + 1023) // 1024  # 32×32 = 1024 threads per group
    
    # Warmup
    shader.run(work_groups, 1, 1)
    ctx.finish()
    
    # Benchmark
    times = []
    for i in range(runs):
        start = time.perf_counter()
        shader.run(work_groups, 1, 1)
        ctx.finish()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # Release
    buf_a.release()
    buf_b.release()
    buf_out.release()
    shader.release()
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    # Throughput: operations per second
    # Each element is one HNS addition operation
    throughput = size / mean_time
    
    print(f"    Mean: {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms")
    print(f"    Throughput: {throughput/1e9:.3f} billion ops/s")
    
    return {
        'operation': 'addition',
        'size': size,
        'runs': runs,
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'throughput_ops_per_sec': throughput,
        'throughput_billion_ops_per_sec': throughput / 1e9
    }


def benchmark_hns_scale(ctx, size: int, runs: int = 20) -> dict:
    """Benchmark pure HNS scaling operations."""
    print(f"\n  Testing HNS Scaling - {size:,} elements")
    
    # Compile shader
    shader = ctx.compute_shader(HNS_SCALE_SHADER)
    
    # Create buffers
    rng = np.random.RandomState(42)
    data_a = rng.randint(0, 1000, size=(size, 4)).astype(np.float32)
    
    buf_a = ctx.buffer(data_a.tobytes())
    buf_out = ctx.buffer(reserve=size * 16)
    
    # Bind buffers
    buf_a.bind_to_storage_buffer(0)
    buf_out.bind_to_storage_buffer(1)
    
    # Set uniform
    shader['u_scale'].value = 1.5
    
    # Calculate work groups
    work_groups = (size + 1023) // 1024
    
    # Warmup
    shader.run(work_groups, 1, 1)
    ctx.finish()
    
    # Benchmark
    times = []
    for i in range(runs):
        start = time.perf_counter()
        shader.run(work_groups, 1, 1)
        ctx.finish()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # Release
    buf_a.release()
    buf_out.release()
    shader.release()
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = size / mean_time
    
    print(f"    Mean: {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms")
    print(f"    Throughput: {throughput/1e9:.3f} billion ops/s")
    
    return {
        'operation': 'scaling',
        'size': size,
        'runs': runs,
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'throughput_ops_per_sec': throughput,
        'throughput_billion_ops_per_sec': throughput / 1e9
    }


def main():
    """Run pure HNS benchmarks matching paper methodology."""
    print("\n" + "="*80)
    print("PURE HNS OPERATIONS BENCHMARK - Paper Validation")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print("="*80)
    
    if not HAS_MODERNGL:
        print("[ERROR] ModernGL required")
        return
    
    # Create OpenGL context
    try:
        ctx = moderngl.create_standalone_context(require=430)
        print(f"\nGPU: {ctx.info['GL_RENDERER']}")
        print(f"OpenGL: {ctx.info['GL_VERSION']}")
    except Exception as e:
        print(f"[ERROR] Failed to create OpenGL context: {e}")
        return
    
    # Test sizes matching paper (Figure 1)
    test_sizes = [
        10_000,      # 10K
        100_000,     # 100K
        1_000_000,   # 1M
        10_000_000,  # 10M (paper's peak test)
    ]
    
    results = {
        'benchmark': 'Pure HNS Operations',
        'timestamp': datetime.now().isoformat(),
        'gpu': ctx.info['GL_RENDERER'],
        'opengl': ctx.info['GL_VERSION'],
        'paper_claims': {
            'addition_10M': '15.9 billion ops/s',
            'scaling_10M': '19.8 billion ops/s'
        },
        'addition_results': [],
        'scaling_results': []
    }
    
    # Test Addition
    print("\n" + "="*80)
    print("HNS ADDITION BENCHMARKS")
    print("="*80)
    
    for size in test_sizes:
        try:
            result = benchmark_hns_add(ctx, size, runs=20)
            results['addition_results'].append(result)
        except Exception as e:
            print(f"  [ERROR] Size {size}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test Scaling
    print("\n" + "="*80)
    print("HNS SCALING BENCHMARKS")
    print("="*80)
    
    for size in test_sizes:
        try:
            result = benchmark_hns_scale(ctx, size, runs=20)
            results['scaling_results'].append(result)
        except Exception as e:
            print(f"  [ERROR] Size {size}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_file = Path(__file__).parent / 'pure_hns_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("PAPER VALIDATION SUMMARY")
    print("="*80)
    
    if results['addition_results']:
        add_10m = [r for r in results['addition_results'] if r['size'] == 10_000_000]
        if add_10m:
            measured = add_10m[0]['throughput_billion_ops_per_sec']
            paper_claim = 15.9
            print(f"\nAddition (10M elements):")
            print(f"  Paper claim: {paper_claim} billion ops/s")
            print(f"  Measured:    {measured:.2f} billion ops/s")
            print(f"  Difference:  {((measured - paper_claim) / paper_claim * 100):+.1f}%")
            
            if abs(measured - paper_claim) / paper_claim < 0.15:
                print(f"  Verdict: ✅ VALIDATED (within 15%)")
            else:
                print(f"  Verdict: ⚠️ DISCREPANCY (>15% difference)")
    
    if results['scaling_results']:
        scale_10m = [r for r in results['scaling_results'] if r['size'] == 10_000_000]
        if scale_10m:
            measured = scale_10m[0]['throughput_billion_ops_per_sec']
            paper_claim = 19.8
            print(f"\nScaling (10M elements):")
            print(f"  Paper claim: {paper_claim} billion ops/s")
            print(f"  Measured:    {measured:.2f} billion ops/s")
            print(f"  Difference:  {((measured - paper_claim) / paper_claim * 100):+.1f}%")
            
            if abs(measured - paper_claim) / paper_claim < 0.15:
                print(f"  Verdict: ✅ VALIDATED (within 15%)")
            else:
                print(f"  Verdict: ⚠️ DISCREPANCY (>15% difference)")
    
    print("\n" + "="*80)
    print("[OK] Pure HNS Benchmark Complete")
    print("="*80)
    
    ctx.release()


if __name__ == '__main__':
    main()
