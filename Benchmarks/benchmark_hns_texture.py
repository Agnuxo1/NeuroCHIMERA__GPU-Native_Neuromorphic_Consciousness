"""
FIXED: Texture-based HNS Benchmark
==================================
Matches Paper Architecture: Uses RGBA32F Textures + Compute Shaders
(Previous attempt used SSBOs, which differs from paper's "Neural State Texture")

Methodology:
1. Create RGBA32F Textures (10M elements approx 3162x3162)
2. Use image2D load/store in Compute Shader
3. Dispatch 2D work groups (32x32 local size)
4. Validate correctness vs CPU
5. Measure throughput
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


# Texture-based HNS Addition Shader
# Matches paper's "Neural State Texture" approach
HNS_TEXTURE_ADD_SHADER = """
#version 430

layout(local_size_x = 32, local_size_y = 32) in;

layout(rgba32f, binding = 0) uniform image2D img_a;
layout(rgba32f, binding = 1) uniform image2D img_b;
layout(rgba32f, binding = 2) uniform image2D img_out;

const float BASE = 1000.0;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(img_a);
    
    if (coords.x >= size.x || coords.y >= size.y) return;
    
    vec4 a = imageLoad(img_a, coords);
    vec4 b = imageLoad(img_b, coords);
    
    // HNS Addition
    float sum0 = a.r + b.r;
    float carry0 = floor(sum0 / BASE);
    vec4 res;
    res.r = mod(sum0, BASE);
    
    float sum1 = a.g + b.g + carry0;
    float carry1 = floor(sum1 / BASE);
    res.g = mod(sum1, BASE);
    
    float sum2 = a.b + b.b + carry1;
    float carry2 = floor(sum2 / BASE);
    res.b = mod(sum2, BASE);
    
    float sum3 = a.a + b.a + carry2;
    res.a = mod(sum3, BASE);
    
    imageStore(img_out, coords, res);
}
"""

def validate_hns_cpu(a, b):
    """CPU Validation."""
    base = 1000.0
    res = np.zeros_like(a)
    carry = 0.0
    
    # R
    s0 = a[:,:,0] + b[:,:,0]
    c0 = np.floor(s0 / base)
    res[:,:,0] = np.mod(s0, base)
    
    # G
    s1 = a[:,:,1] + b[:,:,1] + c0
    c1 = np.floor(s1 / base)
    res[:,:,1] = np.mod(s1, base)
    
    # B
    s2 = a[:,:,2] + b[:,:,2] + c1
    c2 = np.floor(s2 / base)
    res[:,:,2] = np.mod(s2, base)
    
    # A
    s3 = a[:,:,3] + b[:,:,3] + c2
    res[:,:,3] = np.mod(s3, base)
    
    return res

def benchmark_texture_hns(ctx, num_elements, runs=20):
    """Benchmark using Textures."""
    # Calculate texture dimensions (square-ish)
    side = int(np.ceil(np.sqrt(num_elements)))
    # Ensure multiple of 32 for alignment
    side = (side + 31) // 32 * 32
    
    real_elements = side * side
    print(f"\n  Testing Texture HNS - {real_elements:,} elements ({side}x{side})")
    
    # Compile shader
    program = ctx.compute_shader(HNS_TEXTURE_ADD_SHADER)
    
    # Create data
    rng = np.random.RandomState(42)
    data_a = rng.randint(0, 1000, size=(side, side, 4)).astype(np.float32)
    data_b = rng.randint(0, 1000, size=(side, side, 4)).astype(np.float32)
    
    # Create textures
    tex_a = ctx.texture((side, side), 4, data=data_a.tobytes(), dtype='f4')
    tex_b = ctx.texture((side, side), 4, data=data_b.tobytes(), dtype='f4')
    tex_out = ctx.texture((side, side), 4, dtype='f4')
    
    # Bind images
    tex_a.bind_to_image(0, read=True, write=False)
    tex_b.bind_to_image(1, read=True, write=False)
    tex_out.bind_to_image(2, read=False, write=True)
    
    # Work groups
    groups_x = side // 32
    groups_y = side // 32
    
    # Warmup
    print("    Warming up...")
    for _ in range(5):
        program.run(groups_x, groups_y, 1)
    ctx.finish()
    
    # Validation
    print("    Validating...")
    program.run(groups_x, groups_y, 1)
    ctx.finish()
    
    raw = tex_out.read()
    gpu_res = np.frombuffer(raw, dtype=np.float32).reshape(side, side, 4)
    cpu_res = validate_hns_cpu(data_a, data_b)
    
    if not np.allclose(gpu_res, cpu_res, atol=0.1):
        print("    ❌ VALIDATION FAILED")
        diff = np.max(np.abs(gpu_res - cpu_res))
        print(f"    Max diff: {diff}")
        return None
    print("    ✅ Validation Passed")
    
    # Benchmark
    print(f"    Benchmarking {runs} runs...")
    times = []
    
    # Pure GPU timing (batch)
    batch_size = 50
    start = time.perf_counter()
    for _ in range(batch_size):
        program.run(groups_x, groups_y, 1)
    ctx.finish()
    total_time = time.perf_counter() - start
    avg_time = total_time / batch_size
    
    ops_per_sec = real_elements / avg_time
    billions = ops_per_sec / 1e9
    
    print(f"    Time: {avg_time*1000:.3f}ms")
    print(f"    Throughput: {billions:.2f} billion ops/s")
    
    # Cleanup
    tex_a.release()
    tex_b.release()
    tex_out.release()
    
    return billions

def main():
    print("="*60)
    print("TEXTURE-BASED HNS BENCHMARK (Paper Architecture)")
    print("="*60)
    
    if not HAS_MODERNGL:
        print("ModernGL missing")
        return
        
    ctx = moderngl.create_standalone_context(require=430)
    print(f"GPU: {ctx.info['GL_RENDERER']}")
    
    sizes = [100_000, 1_000_000, 10_000_000]
    results = {}
    
    for s in sizes:
        try:
            res = benchmark_texture_hns(ctx, s)
            results[s] = res
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nSUMMARY:")
    for s, r in results.items():
        print(f"{s:,} elements: {r:.2f} B ops/s")

if __name__ == "__main__":
    main()
