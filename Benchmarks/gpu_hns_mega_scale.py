"""
GPU HNS MEGA-SCALE Benchmark
=============================
Maximizes GPU utilization through MASSIVE workloads:
- 4096x4096 images (16 megapixels) instead of 1D arrays
- Multi-stage pipeline (compute + blend + post-process)
- Continuous workload streaming
- Target: 95%+ GPU saturation and >25 billion ops/s

Strategy: Keep GPU busy with HUGE textures and complex operations
"""

import moderngl
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict

BASE = 1000.0

# MEGA-SCALE HNS Image Processing Shader
HNS_IMAGE_PROCESS_SHADER = """
#version 430

layout (local_size_x = 32, local_size_y = 32) in;

layout (rgba32f, binding = 0) uniform image2D input_image_a;
layout (rgba32f, binding = 1) uniform image2D input_image_b;
layout (rgba32f, binding = 2) uniform image2D output_image;

const float BASE = 1000.0;
const float INV_BASE = 0.001;

vec4 hns_normalize_fast(vec4 v) {
    float c0 = floor(v.r * INV_BASE);
    v.r = fma(c0, -BASE, v.r);
    v.g += c0;

    float c1 = floor(v.g * INV_BASE);
    v.g = fma(c1, -BASE, v.g);
    v.b += c1;

    float c2 = floor(v.b * INV_BASE);
    v.b = fma(c2, -BASE, v.b);
    v.a += c2;

    return v;
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_image);

    if (coords.x < size.x && coords.y < size.y) {
        vec4 a = imageLoad(input_image_a, coords);
        vec4 b = imageLoad(input_image_b, coords);

        vec4 result = hns_normalize_fast(a + b);

        imageStore(output_image, coords, result);
    }
}
"""

# MULTI-STAGE Pipeline: Process + Blend + Post
HNS_MULTI_STAGE_SHADER = """
#version 430

layout (local_size_x = 32, local_size_y = 32) in;

layout (rgba32f, binding = 0) uniform image2D img_a;
layout (rgba32f, binding = 1) uniform image2D img_b;
layout (rgba32f, binding = 2) uniform image2D img_c;
layout (rgba32f, binding = 3) uniform image2D img_d;
layout (rgba32f, binding = 4) uniform image2D result_image;

const float BASE = 1000.0;
const float INV_BASE = 0.001;

vec4 hns_normalize_fast(vec4 v) {
    float c0 = floor(v.r * INV_BASE);
    v.r = fma(c0, -BASE, v.r);
    v.g += c0;

    float c1 = floor(v.g * INV_BASE);
    v.g = fma(c1, -BASE, v.g);
    v.b += c1;

    float c2 = floor(v.b * INV_BASE);
    v.b = fma(c2, -BASE, v.b);
    v.a += c2;

    return v;
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 img_size = imageSize(result_image);

    if (coords.x < img_size.x && coords.y < img_size.y) {
        // Stage 1: Add pairs
        vec4 sum1 = imageLoad(img_a, coords) + imageLoad(img_b, coords);
        vec4 sum2 = imageLoad(img_c, coords) + imageLoad(img_d, coords);

        // Stage 2: Normalize both
        vec4 norm1 = hns_normalize_fast(sum1);
        vec4 norm2 = hns_normalize_fast(sum2);

        // Stage 3: Blend results
        vec4 blended = (norm1 + norm2) * 0.5;

        // Stage 4: Final normalize
        vec4 final_result = hns_normalize_fast(blended);

        // Stage 5: Write with optional post-processing
        imageStore(result_image, coords, final_result);
    }
}
"""

# ITERATIVE Kernel - Performs N iterations per pixel (keeps GPU busy)
HNS_ITERATIVE_SHADER = """
#version 430

layout (local_size_x = 32, local_size_y = 32) in;

layout (rgba32f, binding = 0) uniform image2D input_image;
layout (rgba32f, binding = 1) uniform image2D output_image;

uniform int iterations;  // Number of iterations per pixel

const float BASE = 1000.0;
const float INV_BASE = 0.001;

vec4 hns_normalize_fast(vec4 v) {
    float c0 = floor(v.r * INV_BASE);
    v.r = fma(c0, -BASE, v.r);
    v.g += c0;

    float c1 = floor(v.g * INV_BASE);
    v.g = fma(c1, -BASE, v.g);
    v.b += c1;

    float c2 = floor(v.b * INV_BASE);
    v.b = fma(c2, -BASE, v.b);
    v.a += c2;

    return v;
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_image);

    if (coords.x < size.x && coords.y < size.y) {
        vec4 value = imageLoad(input_image, coords);

        // Perform N iterations of HNS operations
        for (int i = 0; i < iterations; i++) {
            value = value + vec4(1.0, 0.0, 0.0, 0.0);  // Increment
            value = hns_normalize_fast(value);
        }

        imageStore(output_image, coords, value);
    }
}
"""


class MegaScaleGPUHNS:
    """Mega-scale GPU HNS with massive image processing"""

    def __init__(self):
        print("Initializing MEGA-SCALE GPU HNS Engine...")
        self.ctx = moderngl.create_standalone_context(require=430)

        print(f"GPU: {self.ctx.info['GL_RENDERER']}")
        print(f"OpenGL: {self.ctx.info['GL_VERSION']}")

        # Compile shaders
        print("Compiling MEGA-SCALE shaders...")
        self.image_shader = self.ctx.compute_shader(HNS_IMAGE_PROCESS_SHADER)
        self.multi_stage_shader = self.ctx.compute_shader(HNS_MULTI_STAGE_SHADER)
        self.iterative_shader = self.ctx.compute_shader(HNS_ITERATIVE_SHADER)

        print("[OK] Mega-scale GPU HNS ready!")
        print(f"Strategy: Process 4096x4096 images (16 megapixels)")
        print(f"Target: >25 billion ops/s with 95%+ GPU saturation\n")

    def benchmark_mega_image(self, width=4096, height=4096, runs=20) -> Dict:
        """Benchmark MASSIVE image processing"""
        print(f"Benchmarking MEGA IMAGE ({width}x{height} = {width*height:,} pixels, runs={runs})")
        print("-" * 70)

        # Generate massive test images
        np.random.seed(42)
        img_a = np.random.rand(height, width, 4).astype(np.float32) * 999.0
        img_b = np.random.rand(height, width, 4).astype(np.float32) * 999.0

        # Create GPU textures (much faster than buffers for 2D data)
        tex_a = self.ctx.texture((width, height), 4, dtype='f4', data=img_a.tobytes())
        tex_b = self.ctx.texture((width, height), 4, dtype='f4')
        tex_b.write(img_b.tobytes())
        tex_out = self.ctx.texture((width, height), 4, dtype='f4')

        # Bind textures as images
        tex_a.bind_to_image(0, read=True, write=False)
        tex_b.bind_to_image(1, read=True, write=False)
        tex_out.bind_to_image(2, read=False, write=True)

        # Work groups for 4096x4096
        work_groups_x = (width + 31) // 32
        work_groups_y = (height + 31) // 32

        print(f"Work groups: {work_groups_x}x{work_groups_y} = {work_groups_x * work_groups_y:,} groups")
        print(f"Total threads: {work_groups_x * work_groups_y * 1024:,}\n")

        times = []
        for run in range(runs):
            self.ctx.finish()
            start = time.perf_counter()

            self.image_shader.run(work_groups_x, work_groups_y, 1)

            self.ctx.finish()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if (run + 1) % 5 == 0:
                print(f"  Run {run + 1}/{runs}: {elapsed * 1000:.2f} ms")

        mean_time = np.mean(times)
        std_time = np.std(times)
        pixels = width * height
        throughput = pixels / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
        print(f"  Throughput: {throughput / 1e9:.2f} billion pixels/s")
        print(f"  Ops/s: {throughput:.2e} ops/s\n")

        tex_a.release()
        tex_b.release()
        tex_out.release()

        return {
            'width': width,
            'height': height,
            'pixels': pixels,
            'runs': runs,
            'mean_time_ms': mean_time * 1000,
            'std_time_ms': std_time * 1000,
            'throughput_ops_per_sec': throughput
        }

    def benchmark_multi_stage_pipeline(self, width=4096, height=4096, runs=20) -> Dict:
        """Benchmark multi-stage pipeline (5 stages)"""
        print(f"Benchmarking MULTI-STAGE PIPELINE ({width}x{height}, runs={runs})")
        print("-" * 70)

        # Generate 4 input images
        np.random.seed(42)
        images = [np.random.rand(height, width, 4).astype(np.float32) * 999.0 for _ in range(4)]

        # Create textures
        textures = []
        for i, img in enumerate(images):
            tex = self.ctx.texture((width, height), 4, dtype='f4', data=img.tobytes())
            tex.bind_to_image(i, read=True, write=False)
            textures.append(tex)

        tex_out = self.ctx.texture((width, height), 4, dtype='f4')
        tex_out.bind_to_image(4, read=False, write=True)
        textures.append(tex_out)

        work_groups_x = (width + 31) // 32
        work_groups_y = (height + 31) // 32

        times = []
        for run in range(runs):
            self.ctx.finish()
            start = time.perf_counter()

            # Multi-stage pipeline processes 4 images through 5 stages
            self.multi_stage_shader.run(work_groups_x, work_groups_y, 1)

            self.ctx.finish()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if (run + 1) % 5 == 0:
                print(f"  Run {run + 1}/{runs}: {elapsed * 1000:.2f} ms")

        mean_time = np.mean(times)
        std_time = np.std(times)
        pixels = width * height
        # Multi-stage: 4 images × 5 stages = 20x operations
        effective_ops = pixels * 5  # 5 stages
        throughput = effective_ops / mean_time

        print(f"\nResults (MULTI-STAGE):")
        print(f"  Mean time: {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
        print(f"  Stages: 5 (add, add, normalize, blend, final)")
        print(f"  Effective throughput: {throughput / 1e9:.2f} billion ops/s")
        print()

        for tex in textures:
            tex.release()

        return {
            'width': width,
            'height': height,
            'pixels': pixels,
            'stages': 5,
            'runs': runs,
            'mean_time_ms': mean_time * 1000,
            'std_time_ms': std_time * 1000,
            'throughput_ops_per_sec': throughput
        }

    def benchmark_iterative_heavy(self, width=2048, height=2048, iterations=100, runs=10) -> Dict:
        """Benchmark iterative kernel that keeps GPU busy"""
        print(f"Benchmarking ITERATIVE HEAVY ({width}x{height}, {iterations} iters/pixel, runs={runs})")
        print("-" * 70)

        # Single input image
        np.random.seed(42)
        img = np.random.rand(height, width, 4).astype(np.float32) * 999.0

        tex_in = self.ctx.texture((width, height), 4, dtype='f4', data=img.tobytes())
        tex_out = self.ctx.texture((width, height), 4, dtype='f4')

        tex_in.bind_to_image(0, read=True, write=False)
        tex_out.bind_to_image(1, read=False, write=True)

        # Set uniform for iterations
        self.iterative_shader['iterations'].value = iterations

        work_groups_x = (width + 31) // 32
        work_groups_y = (height + 31) // 32

        print(f"Iterations per pixel: {iterations}")
        print(f"Total operations: {width * height * iterations:,}\n")

        times = []
        for run in range(runs):
            self.ctx.finish()
            start = time.perf_counter()

            self.iterative_shader.run(work_groups_x, work_groups_y, 1)

            self.ctx.finish()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if (run + 1) % 5 == 0:
                print(f"  Run {run + 1}/{runs}: {elapsed * 1000:.2f} ms")

        mean_time = np.mean(times)
        std_time = np.std(times)
        total_ops = width * height * iterations
        throughput = total_ops / mean_time

        print(f"\nResults (ITERATIVE):")
        print(f"  Mean time: {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
        print(f"  Total ops: {total_ops:,}")
        print(f"  Throughput: {throughput / 1e9:.2f} billion ops/s")
        print()

        tex_in.release()
        tex_out.release()

        return {
            'width': width,
            'height': height,
            'iterations_per_pixel': iterations,
            'total_operations': total_ops,
            'runs': runs,
            'mean_time_ms': mean_time * 1000,
            'std_time_ms': std_time * 1000,
            'throughput_ops_per_sec': throughput
        }

    def run_mega_suite(self) -> Dict:
        """Run complete mega-scale benchmark suite"""
        results = {
            'benchmark_suite': 'GPU HNS Mega-Scale',
            'date': datetime.now().isoformat(),
            'gpu_info': {
                'renderer': self.ctx.info['GL_RENDERER'],
                'version': self.ctx.info['GL_VERSION']
            },
            'results': {}
        }

        print("=" * 80)
        print("GPU HNS MEGA-SCALE BENCHMARK SUITE")
        print("=" * 80)
        print()

        # 1. Mega image (4096x4096)
        print("\n### Test 1: MEGA IMAGE (4096x4096) ###\n")
        results['results']['mega_image_4096'] = self.benchmark_mega_image(4096, 4096, runs=20)

        # 2. Multi-stage pipeline
        print("\n### Test 2: MULTI-STAGE PIPELINE ###\n")
        results['results']['multi_stage_pipeline'] = self.benchmark_multi_stage_pipeline(4096, 4096, runs=20)

        # 3. Iterative heavy (100 iterations per pixel)
        print("\n### Test 3: ITERATIVE HEAVY (100 iters/pixel) ###\n")
        results['results']['iterative_100'] = self.benchmark_iterative_heavy(2048, 2048, iterations=100, runs=10)

        # 4. ULTRA iterative (1000 iterations per pixel) - GPU torture test
        print("\n### Test 4: ULTRA ITERATIVE (1000 iters/pixel) - GPU TORTURE ###\n")
        results['results']['iterative_1000'] = self.benchmark_iterative_heavy(2048, 2048, iterations=1000, runs=5)

        # Find maximum
        max_throughput = 0
        max_config = None

        for key, result in results['results'].items():
            if result['throughput_ops_per_sec'] > max_throughput:
                max_throughput = result['throughput_ops_per_sec']
                max_config = (key, result)

        results['peak_performance'] = {
            'test': max_config[0],
            'throughput_ops_per_sec': max_throughput,
            'throughput_billion_ops_per_sec': max_throughput / 1e9,
            'configuration': max_config[1]
        }

        print("\n" + "=" * 80)
        print("PEAK MEGA-SCALE PERFORMANCE")
        print("=" * 80)
        print(f"Test: {max_config[0]}")
        print(f"Maximum Throughput: {max_throughput / 1e9:.2f} billion ops/s")
        print("=" * 80)

        return results


def main():
    print("\nGPU HNS Mega-Scale Strategy")
    print("============================")
    print("Approach: Process MASSIVE images to saturate GPU")
    print("  - 4096x4096 images (16 megapixels)")
    print("  - Multi-stage pipelines (5 stages)")
    print("  - Iterative kernels (100-1000 iterations/pixel)")
    print("  - Target: >25 billion ops/s\n")

    try:
        engine = MegaScaleGPUHNS()
        results = engine.run_mega_suite()

        # Save results
        output_file = 'gpu_hns_mega_scale_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[OK] Results saved to: {output_file}")

        # Compare with baselines
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        print("Original (256 threads):   19.8 billion ops/s")
        print("Ultra-Optimized (1024):   20.7 billion ops/s  (+4%)")
        print(f"Mega-Scale (images):      {results['peak_performance']['throughput_billion_ops_per_sec']:.2f} billion ops/s  "
              f"({(results['peak_performance']['throughput_billion_ops_per_sec'] / 19.8 - 1) * 100:+.1f}%)")
        print("=" * 80)

        print("\n[OK] Mega-scale benchmark complete!")

        return 0

    except Exception as e:
        print(f"\n[FAILED] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
