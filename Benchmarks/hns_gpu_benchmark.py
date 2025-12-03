"""
GPU BENCHMARK: HNS 100% on GPU using OpenGL/GLSL
=================================================
This benchmark executes HNS completely on GPU using GLSL shaders
and compares real performance with standard float on GPU.

Requirements:
- moderngl
- numpy
- A GPU compatible with OpenGL 3.3+
"""

import moderngl
import numpy as np
import time
import sys
from typing import Dict, Tuple, List

# ===================================================================
# SHADERS GLSL
# ===================================================================

# Include hns_core.glsl (without #version, added in each shader)
HNS_CORE_GLSL = """
// =================================================================
// HNS LIBRARY (HIERARCHICAL NUMBER SYSTEM) - VESELOV/ANGULO
// GPU/GLSL Implementation v1.0
// =================================================================

const float BASE = 1000.0; 
const float INV_BASE = 0.001;

#define HNumber vec4

HNumber hns_normalize(HNumber n) {
    HNumber res = n;
    
    float carry0 = floor(res.r * INV_BASE);
    res.r = res.r - (carry0 * BASE);
    res.g += carry0;
    
    float carry1 = floor(res.g * INV_BASE);
    res.g = res.g - (carry1 * BASE);
    res.b += carry1;
    
    float carry2 = floor(res.b * INV_BASE);
    res.b = res.b - (carry2 * BASE);
    res.a += carry2;
    
    return res;
}

HNumber hns_add(HNumber a, HNumber b) {
    HNumber sum = a + b;
    return hns_normalize(sum);
}

HNumber hns_scale(HNumber a, float scalar) {
    HNumber scaled = a * scalar;
    return hns_normalize(scaled);
}

float hns_to_float(HNumber n) {
    return n.r + 
           (n.g * BASE) + 
           (n.b * BASE * BASE) + 
           (n.a * BASE * BASE * BASE);
}
"""

# Shader for HNS addition on GPU
HNS_ADD_SHADER = """
#version 330
""" + HNS_CORE_GLSL + """
out vec4 fragColor;

uniform sampler2D texture_a;
uniform sampler2D texture_b;
uniform vec2 resolution;

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    HNumber a = texture(texture_a, uv);
    HNumber b = texture(texture_b, uv);
    HNumber result = hns_add(a, b);
    fragColor = result;
}
"""

# Shader for standard float addition on GPU (comparison)
FLOAT_ADD_SHADER = """
#version 330

out vec4 fragColor;

uniform sampler2D texture_a;
uniform sampler2D texture_b;
uniform vec2 resolution;

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    float a = texture(texture_a, uv).r;
    float b = texture(texture_b, uv).r;
    float result = a + b;
    fragColor = vec4(result, result, result, 1.0);
}
"""

# Shader for HNS scaling
HNS_SCALE_SHADER = """
#version 330
""" + HNS_CORE_GLSL + """
out vec4 fragColor;

uniform sampler2D texture_a;
uniform float scalar;
uniform vec2 resolution;

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    HNumber a = texture(texture_a, uv);
    HNumber result = hns_scale(a, scalar);
    fragColor = result;
}
"""

# Shader for standard float scaling
FLOAT_SCALE_SHADER = """
#version 330

out vec4 fragColor;

uniform sampler2D texture_a;
uniform float scalar;
uniform vec2 resolution;

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    float a = texture(texture_a, uv).r;
    float result = a * scalar;
    fragColor = vec4(result, result, result, 1.0);
}
"""

# Shader for HNS accumulation (neural network simulation)
HNS_ACCUMULATE_SHADER = """
#version 330
""" + HNS_CORE_GLSL + """
in vec2 fragCoord;
out vec4 fragColor;

uniform sampler2D texture_accumulator;
uniform sampler2D texture_input;
uniform vec2 resolution;

void main() {
    vec2 uv = fragCoord;
    HNumber accumulator = texture(texture_accumulator, uv);
    HNumber input_val = texture(texture_input, uv);
    HNumber result = hns_add(accumulator, input_val);
    fragColor = result;
}
"""

# Shader for standard float accumulation
FLOAT_ACCUMULATE_SHADER = """
#version 330

in vec2 fragCoord;
out vec4 fragColor;

uniform sampler2D texture_accumulator;
uniform sampler2D texture_input;
uniform vec2 resolution;

void main() {
    vec2 uv = fragCoord;
    float accumulator = texture(texture_accumulator, uv).r;
    float input_val = texture(texture_input, uv).r;
    float result = accumulator + input_val;
    fragColor = vec4(result, result, result, 1.0);
}
"""

# Simple vertex shader (common for all)
VERTEX_SHADER = """
#version 330

in vec2 in_vert;

void main() {
    gl_Position = vec4(in_vert * 2.0 - 1.0, 0.0, 1.0);
}
"""

# ===================================================================
# FUNCIONES DE UTILIDAD
# ===================================================================

def create_context():
    """Creates an OpenGL context using moderngl."""
    try:
        ctx = moderngl.create_context(standalone=True, require=330)
        return ctx
    except Exception as e:
        print(f"[ERROR] Error creating OpenGL context: {e}")
        print("Make sure you have a GPU compatible with OpenGL 3.3+")
        sys.exit(1)

def create_program(ctx, vertex_shader, fragment_shader):
    """Creates a shader program."""
    try:
        prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        return prog
    except Exception as e:
        print(f"[ERROR] Error compiling shader: {e}")
        raise

def create_texture(ctx, width, height, data=None):
    """Creates an RGBA32F texture."""
    if data is None:
        texture = ctx.texture((width, height), 4, dtype='f4')
    else:
        texture = ctx.texture((width, height), 4, data=data, dtype='f4')
    return texture

def create_framebuffer(ctx, texture):
    """Creates a framebuffer."""
    fbo = ctx.framebuffer([texture])
    return fbo

def float_to_hns_gpu(value: float) -> Tuple[float, float, float, float]:
    """Converts a float to HNS representation (RGBA)."""
    BASE = 1000.0
    if value == 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    
    a = np.floor(value / (BASE ** 3))
    remainder = value - (a * BASE ** 3)
    b = np.floor(remainder / (BASE ** 2))
    remainder = remainder - (b * BASE ** 2)
    g = np.floor(remainder / BASE)
    r = remainder - (g * BASE)
    
    return (float(r), float(g), float(b), float(a))

def hns_to_float_gpu(r: float, g: float, b: float, a: float) -> float:
    """Converts HNS (RGBA) to float."""
    BASE = 1000.0
    return r + (g * BASE) + (b * BASE * BASE) + (a * BASE * BASE * BASE)

# ===================================================================
# BENCHMARKS GPU
# ===================================================================

def benchmark_gpu_add(ctx, width=1024, height=1024, iterations=100):
    """GPU addition benchmark: HNS vs Float."""
    print("\n" + "="*80)
    print("GPU BENCHMARK: ADDITION (HNS vs Float)")
    print("="*80)
    print(f"Resolution: {width}x{height} ({width*height:,} pixels)")
    print(f"Iterations: {iterations}")
    
    # Prepare test data
    # Create textures with test numbers
    test_values_a = np.random.uniform(0, 1000000, (height, width)).astype(np.float32)
    test_values_b = np.random.uniform(0, 1000000, (height, width)).astype(np.float32)
    
    # For HNS: convert to RGBA format
    hns_data_a = np.zeros((height, width, 4), dtype=np.float32)
    hns_data_b = np.zeros((height, width, 4), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = float_to_hns_gpu(test_values_a[y, x])
            hns_data_a[y, x] = [r, g, b, a]
            r, g, b, a = float_to_hns_gpu(test_values_b[y, x])
            hns_data_b[y, x] = [r, g, b, a]
    
    # Create textures
    # For float, we need to convert to RGBA (repeat value in all 4 channels)
    float_data_a = np.stack([test_values_a] * 4, axis=2)
    float_data_b = np.stack([test_values_b] * 4, axis=2)
    
    hns_tex_a = create_texture(ctx, width, height, hns_data_a.tobytes())
    hns_tex_b = create_texture(ctx, width, height, hns_data_b.tobytes())
    float_tex_a = create_texture(ctx, width, height, float_data_a.tobytes())
    float_tex_b = create_texture(ctx, width, height, float_data_b.tobytes())
    
    # Create output textures (RGBA for HNS, only R for float but we use RGBA)
    hns_output = create_texture(ctx, width, height)
    float_output = create_texture(ctx, width, height)
    
    # Create framebuffers
    hns_fbo = create_framebuffer(ctx, hns_output)
    float_fbo = create_framebuffer(ctx, float_output)
    
    # Create shader programs
    hns_prog = create_program(ctx, VERTEX_SHADER, HNS_ADD_SHADER)
    float_prog = create_program(ctx, VERTEX_SHADER, FLOAT_ADD_SHADER)
    
    # Create geometry (full quad)
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype=np.float32)
    
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(hns_prog, vbo, 'in_vert')
    
    # Configure uniforms
    resolution = (width, height)
    hns_prog['resolution'].value = resolution
    float_prog['resolution'].value = resolution
    
    # Benchmark HNS
    hns_prog['texture_a'].value = 0
    hns_prog['texture_b'].value = 1
    
    ctx.enable(moderngl.NOTHING)
    hns_fbo.use()
    
    hns_tex_a.use(0)
    hns_tex_b.use(1)
    
    # Warmup
    vao.render(moderngl.TRIANGLE_STRIP)
    ctx.finish()
    
    # Measure HNS time
    start = time.perf_counter()
    for _ in range(iterations):
        vao.render(moderngl.TRIANGLE_STRIP)
    ctx.finish()
    hns_time = time.perf_counter() - start
    
    # Benchmark Float
    float_prog['texture_a'].value = 0
    float_prog['texture_b'].value = 1
    
    float_fbo.use()
    float_tex_a.use(0)
    float_tex_b.use(1)
    
    vao_float = ctx.simple_vertex_array(float_prog, vbo, 'in_vert')
    
    # Warmup
    vao_float.render(moderngl.TRIANGLE_STRIP)
    ctx.finish()
    
    # Measure Float time
    start = time.perf_counter()
    for _ in range(iterations):
        vao_float.render(moderngl.TRIANGLE_STRIP)
    ctx.finish()
    float_time = time.perf_counter() - start
    
    # Read results for verification
    hns_result = np.frombuffer(hns_output.read(), dtype=np.float32).reshape((height, width, 4))
    float_result = np.frombuffer(float_output.read(), dtype=np.float32).reshape((height, width, 4))
    
    # Calculate metrics
    total_pixels = width * height
    total_ops = total_pixels * iterations
    
    hns_ops_per_sec = total_ops / hns_time
    float_ops_per_sec = total_ops / float_time
    overhead = hns_time / float_time
    
    print(f"\nResults:")
    print(f"  HNS:   {hns_time*1000:.4f}ms ({hns_ops_per_sec/1e6:.2f}M ops/s)")
    print(f"  Float: {float_time*1000:.4f}ms ({float_ops_per_sec/1e6:.2f}M ops/s)")
    print(f"  Overhead: {overhead:.2f}x")
    
    if overhead < 1.0:
        print(f"  [OK] HNS is {1.0/overhead:.2f}x FASTER than float")
    else:
        print(f"  [WARN] HNS is {overhead:.2f}x slower than float")
    
    # Cleanup
    hns_tex_a.release()
    hns_tex_b.release()
    float_tex_a.release()
    float_tex_b.release()
    hns_output.release()
    float_output.release()
    hns_fbo.release()
    float_fbo.release()
    
    return {
        'hns_time': hns_time,
        'float_time': float_time,
        'overhead': overhead,
        'hns_ops_per_sec': hns_ops_per_sec,
        'float_ops_per_sec': float_ops_per_sec
    }

def benchmark_gpu_scale(ctx, width=1024, height=1024, iterations=100):
    """GPU scaling benchmark: HNS vs Float."""
    print("\n" + "="*80)
    print("GPU BENCHMARK: SCALING (HNS vs Float)")
    print("="*80)
    print(f"Resolution: {width}x{height} ({width*height:,} pixels)")
    print(f"Iterations: {iterations}")
    
    # Prepare data
    test_values = np.random.uniform(0, 1000000, (height, width)).astype(np.float32)
    scalar = 2.5
    
    # HNS
    hns_data = np.zeros((height, width, 4), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            r, g, b, a = float_to_hns_gpu(test_values[y, x])
            hns_data[y, x] = [r, g, b, a]
    
    # Create textures
    float_data = np.stack([test_values] * 4, axis=2)
    hns_tex = create_texture(ctx, width, height, hns_data.tobytes())
    float_tex = create_texture(ctx, width, height, float_data.tobytes())
    
    hns_output = create_texture(ctx, width, height)
    float_output = create_texture(ctx, width, height)
    
    hns_fbo = create_framebuffer(ctx, hns_output)
    float_fbo = create_framebuffer(ctx, float_output)
    
    # Programs
    hns_prog = create_program(ctx, VERTEX_SHADER, HNS_SCALE_SHADER)
    float_prog = create_program(ctx, VERTEX_SHADER, FLOAT_SCALE_SHADER)
    
    vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(hns_prog, vbo, 'in_vert')
    
    resolution = (width, height)
    hns_prog['resolution'].value = resolution
    hns_prog['scalar'].value = scalar
    float_prog['resolution'].value = resolution
    float_prog['scalar'].value = scalar
    
    # HNS
    hns_fbo.use()
    hns_tex.use(0)
    vao.render(moderngl.TRIANGLE_STRIP)
    ctx.finish()
    
    start = time.perf_counter()
    for _ in range(iterations):
        vao.render(moderngl.TRIANGLE_STRIP)
    ctx.finish()
    hns_time = time.perf_counter() - start
    
    # Float
    float_fbo.use()
    float_tex.use(0)
    vao_float = ctx.simple_vertex_array(float_prog, vbo, 'in_vert')
    vao_float.render(moderngl.TRIANGLE_STRIP)
    ctx.finish()
    
    start = time.perf_counter()
    for _ in range(iterations):
        vao_float.render(moderngl.TRIANGLE_STRIP)
    ctx.finish()
    float_time = time.perf_counter() - start
    
    total_ops = width * height * iterations
    overhead = hns_time / float_time
    
    print(f"\nResults:")
    print(f"  HNS:   {hns_time*1000:.4f}ms ({total_ops/hns_time/1e6:.2f}M ops/s)")
    print(f"  Float: {float_time*1000:.4f}ms ({total_ops/float_time/1e6:.2f}M ops/s)")
    print(f"  Overhead: {overhead:.2f}x")
    
    # Cleanup
    hns_tex.release()
    float_tex.release()
    hns_output.release()
    float_output.release()
    hns_fbo.release()
    float_fbo.release()
    
    return {
        'hns_time': hns_time,
        'float_time': float_time,
        'overhead': overhead
    }

def benchmark_gpu_precision(ctx, width=1024, height=1024):
    """GPU precision benchmark: cases where float32 fails."""
    print("\n" + "="*80)
    print("GPU BENCHMARK: PRECISION (HNS vs Float32)")
    print("="*80)
    print(f"Resolution: {width}x{height}")
    
    # Test cases where float32 may lose precision
    test_cases = [
        ("999,999 + 1", 999999.0, 1.0),
        ("9,999,999 + 1", 9999999.0, 1.0),
        ("1234567.89 + 0.01", 1234567.89, 0.01),
    ]
    
    results = []
    
    for name, a_val, b_val in test_cases:
        # Create textures with the same value in all pixels
        hns_a_data = np.zeros((height, width, 4), dtype=np.float32)
        hns_b_data = np.zeros((height, width, 4), dtype=np.float32)
        r_a, g_a, b_a, a_a = float_to_hns_gpu(a_val)
        r_b, g_b, b_b, a_b = float_to_hns_gpu(b_val)
        
        hns_a_data[:, :] = [r_a, g_a, b_a, a_a]
        hns_b_data[:, :] = [r_b, g_b, b_b, a_b]
        
        float_a_data = np.stack([np.full((height, width), a_val, dtype=np.float32)] * 4, axis=2)
        float_b_data = np.stack([np.full((height, width), b_val, dtype=np.float32)] * 4, axis=2)
        
        # Create textures and execute
        hns_tex_a = create_texture(ctx, width, height, hns_a_data.tobytes())
        hns_tex_b = create_texture(ctx, width, height, hns_b_data.tobytes())
        float_tex_a = create_texture(ctx, width, height, float_a_data.tobytes())
        float_tex_b = create_texture(ctx, width, height, float_b_data.tobytes())
        
        hns_output = create_texture(ctx, width, height)
        float_output = create_texture(ctx, width, height)
        
        hns_fbo = create_framebuffer(ctx, hns_output)
        float_fbo = create_framebuffer(ctx, float_output)
        
        hns_prog = create_program(ctx, VERTEX_SHADER, HNS_ADD_SHADER)
        float_prog = create_program(ctx, VERTEX_SHADER, FLOAT_ADD_SHADER)
        
        vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        vbo = ctx.buffer(vertices.tobytes())
        
        resolution = (width, height)
        hns_prog['resolution'].value = resolution
        float_prog['resolution'].value = resolution
        
        # HNS
        hns_fbo.use()
        hns_tex_a.use(0)
        hns_tex_b.use(1)
        hns_prog['texture_a'].value = 0
        hns_prog['texture_b'].value = 1
        vao = ctx.simple_vertex_array(hns_prog, vbo, 'in_vert')
        vao.render(moderngl.TRIANGLE_STRIP)
        ctx.finish()
        
        # Float
        float_fbo.use()
        float_tex_a.use(0)
        float_tex_b.use(1)
        float_prog['texture_a'].value = 0
        float_prog['texture_b'].value = 1
        vao_float = ctx.simple_vertex_array(float_prog, vbo, 'in_vert')
        vao_float.render(moderngl.TRIANGLE_STRIP)
        ctx.finish()
        
        # Read results
        hns_result = np.frombuffer(hns_output.read(), dtype=np.float32).reshape((height, width, 4))
        float_result = np.frombuffer(float_output.read(), dtype=np.float32).reshape((height, width, 4))
        
        # Convert HNS to float
        hns_value = hns_to_float_gpu(hns_result[0, 0, 0], hns_result[0, 0, 1], 
                                     hns_result[0, 0, 2], hns_result[0, 0, 3])
        float_value = float_result[0, 0, 0]
        expected = a_val + b_val
        
        hns_error = abs(hns_value - expected)
        float_error = abs(float_value - expected)
        
        results.append({
            'name': name,
            'expected': expected,
            'hns_value': hns_value,
            'float_value': float_value,
            'hns_error': hns_error,
            'float_error': float_error,
            'hns_better': hns_error < float_error
        })
        
        print(f"\n{name}:")
        print(f"  Expected: {expected}")
        print(f"  HNS:   {hns_value} (Error: {hns_error:.2e})")
        print(f"  Float: {float_value} (Error: {float_error:.2e})")
        if hns_error < float_error:
            print(f"  [OK] HNS is more precise")
        elif float_error < hns_error:
            print(f"  [WARN] Float is more precise")
        else:
            print(f"  [-] Same precision")
        
        # Cleanup
        hns_tex_a.release()
        hns_tex_b.release()
        float_tex_a.release()
        float_tex_b.release()
        hns_output.release()
        float_output.release()
        hns_fbo.release()
        float_fbo.release()
    
    return results

# ===================================================================
# MAIN
# ===================================================================

def main():
    """Executes all GPU benchmarks."""
    print("\n" + "="*80)
    print("GPU BENCHMARK: HNS 100% on GPU (OpenGL/GLSL)")
    print("="*80)
    print("\nThis benchmark executes HNS completely on GPU using GLSL shaders")
    print("and compares real performance with standard float.\n")
    
    try:
        # Create context
        print("Initializing OpenGL context...")
        ctx = create_context()
        print(f"[OK] Context created: {ctx.info['GL_VENDOR']} - {ctx.info['GL_RENDERER']}")
        print(f"   OpenGL {ctx.info['GL_VERSION']}")
        
        # Execute benchmarks
        results = {}
        
        # Precision benchmark (small, fast)
        results['precision'] = benchmark_gpu_precision(ctx, width=512, height=512)
        
        # Addition benchmark
        results['add'] = benchmark_gpu_add(ctx, width=1024, height=1024, iterations=100)
        
        # Scaling benchmark
        results['scale'] = benchmark_gpu_scale(ctx, width=1024, height=1024, iterations=100)
        
        # Summary
        print("\n" + "="*80)
        print("GPU BENCHMARK SUMMARY")
        print("="*80)
        
        if 'add' in results:
            print(f"\nAddition:")
            print(f"  HNS Overhead: {results['add']['overhead']:.2f}x")
            print(f"  HNS Throughput: {results['add']['hns_ops_per_sec']/1e6:.2f}M ops/s")
            print(f"  Float Throughput: {results['add']['float_ops_per_sec']/1e6:.2f}M ops/s")
        
        if 'scale' in results:
            print(f"\nScaling:")
            print(f"  HNS Overhead: {results['scale']['overhead']:.2f}x")
        
        if 'precision' in results:
            precision_wins = sum(1 for r in results['precision'] if r.get('hns_better', False))
            total = len(results['precision'])
            print(f"\nPrecision:")
            print(f"  HNS more precise in {precision_wins}/{total} cases")
        
        print("\n[OK] GPU benchmark completed")
        
        # Cleanup
        ctx.release()
        
    except KeyboardInterrupt:
        print("\n\n[WARN] Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] ERROR during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

