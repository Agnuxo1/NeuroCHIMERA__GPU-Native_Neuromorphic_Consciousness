"""
GPU HNS Precision Certification
===============================
Certifies that the HNS precision benefits (large numbers, accumulation)
are preserved when running on the GPU.

Tests:
1. Large Number Addition (e.g., 999,999 + 1)
2. Accumulative Precision (1M small additions)
"""

import moderngl
import numpy as np
import time
import json
import struct

# HNS Addition Shader (Same as in core)
HNS_ADD_SHADER = """
#version 430

layout (local_size_x = 1) in;

layout (std430, binding = 0) buffer InputA {
    vec4 a[];
};

layout (std430, binding = 1) buffer InputB {
    vec4 b[];
};

layout (std430, binding = 2) buffer Output {
    vec4 result[];
};

const float BASE = 1000.0;
const float INV_BASE = 0.001;

vec4 hns_normalize(vec4 v) {
    float carry0 = floor(v.r * INV_BASE);
    v.r = v.r - (carry0 * BASE);
    v.g += carry0;

    float carry1 = floor(v.g * INV_BASE);
    v.g = v.g - (carry1 * BASE);
    v.b += carry1;

    float carry2 = floor(v.b * INV_BASE);
    v.b = v.b - (carry2 * BASE);
    v.a += carry2;

    return v;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < a.length()) {
        vec4 raw_sum = a[idx] + b[idx];
        result[idx] = hns_normalize(raw_sum);
    }
}
"""

# HNS Accumulation Shader
HNS_ACCUMULATE_SHADER = """
#version 430

layout (local_size_x = 1) in;

layout (std430, binding = 0) buffer Input {
    vec4 start_val;
    vec4 increment;
    int iterations;
};

layout (std430, binding = 1) buffer Output {
    vec4 result;
};

const float BASE = 1000.0;
const float INV_BASE = 0.001;

vec4 hns_normalize(vec4 v) {
    float carry0 = floor(v.r * INV_BASE);
    v.r = v.r - (carry0 * BASE);
    v.g += carry0;

    float carry1 = floor(v.g * INV_BASE);
    v.g = v.g - (carry1 * BASE);
    v.b += carry1;

    float carry2 = floor(v.b * INV_BASE);
    v.b = v.b - (carry2 * BASE);
    v.a += carry2;

    return v;
}

void main() {
    vec4 acc = start_val;
    vec4 inc = increment;
    
    for(int i=0; i<iterations; i++) {
        acc = hns_normalize(acc + inc);
    }
    
    result = acc;
}
"""

def float_to_hns(val):
    """Convert float to HNS vec4 (CPU helper)."""
    base = 1000.0
    r = val % base
    val = val // base
    g = val % base
    val = val // base
    b = val % base
    val = val // base
    a = val
    return np.array([r, g, b, a], dtype=np.float32)

def hns_to_float(vec):
    """Convert HNS vec4 to float (CPU helper)."""
    base = 1000.0
    return vec[0] + vec[1]*base + vec[2]*base**2 + vec[3]*base**3

class GPUPrecisionCertifier:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=430)
        self.add_prog = self.ctx.compute_shader(HNS_ADD_SHADER)
        self.acc_prog = self.ctx.compute_shader(HNS_ACCUMULATE_SHADER)
        print(f"GPU Context: {self.ctx.info['GL_RENDERER']}")

    def test_large_numbers(self):
        print("\nTest 1: Large Number Addition (GPU)")
        print("-" * 40)
        
        # Test Case: 999,999 + 1 = 1,000,000
        # HNS: [999, 999, 0, 0] + [1, 0, 0, 0] -> [0, 0, 1, 0] (Normalized)
        
        a_val = 999999.0
        b_val = 1.0
        expected = 1000000.0
        
        a_hns = float_to_hns(a_val)
        b_hns = float_to_hns(b_val)
        
        # Buffers
        buf_a = self.ctx.buffer(a_hns.tobytes())
        buf_b = self.ctx.buffer(b_hns.tobytes())
        buf_res = self.ctx.buffer(reserve=16)
        
        buf_a.bind_to_storage_buffer(0)
        buf_b.bind_to_storage_buffer(1)
        buf_res.bind_to_storage_buffer(2)
        
        self.add_prog.run(1, 1, 1)
        self.ctx.finish()
        
        res_bytes = buf_res.read()
        res_hns = np.frombuffer(res_bytes, dtype=np.float32)
        res_float = hns_to_float(res_hns)
        
        print(f"  {a_val:,.0f} + {b_val:,.0f}")
        print(f"  Expected: {expected:,.0f}")
        print(f"  GPU Result (HNS): {res_hns}")
        print(f"  GPU Result (Float): {res_float:,.0f}")
        
        if abs(res_float - expected) < 0.001:
            print("  [PASS] Precision Preserved")
            return True
        else:
            print("  [FAIL] Precision Lost")
            return False

    def test_accumulation(self):
        print("\nTest 2: Accumulative Precision (GPU)")
        print("-" * 40)
        
        # Accumulate 0.001, 1,000,000 times -> 1000.0
        # Float32 usually drifts significantly here if not careful, 
        # but HNS separates small increments into the 'r' channel.
        
        iterations = 1_000_000
        increment = 0.001
        expected = iterations * increment
        
        # Struct: vec4 start, vec4 inc, int iterations
        # 4 floats + 4 floats + 1 int = 36 bytes? No, std430 alignment rules.
        # vec4 is 16 bytes aligned.
        # start (16), inc (16), iterations (4) -> 36 bytes.
        
        start_hns = np.array([0,0,0,0], dtype=np.float32)
        inc_hns = np.array([increment, 0,0,0], dtype=np.float32) # Purely in lowest channel
        
        # Pack data
        data = start_hns.tobytes() + inc_hns.tobytes() + struct.pack('i', iterations)
        
        buf_in = self.ctx.buffer(data)
        buf_out = self.ctx.buffer(reserve=16)
        
        buf_in.bind_to_storage_buffer(0)
        buf_out.bind_to_storage_buffer(1)
        
        t0 = time.perf_counter()
        self.acc_prog.run(1, 1, 1)
        self.ctx.finish()
        dt = time.perf_counter() - t0
        
        res_hns = np.frombuffer(buf_out.read(), dtype=np.float32)
        res_float = hns_to_float(res_hns)
        
        print(f"  Accumulating {increment} x {iterations:,}")
        print(f"  Expected: {expected:.6f}")
        print(f"  GPU Result: {res_float:.6f}")
        print(f"  Time: {dt*1000:.2f} ms")
        
        error = abs(res_float - expected)
        print(f"  Error: {error:.2e}")
        
        if error < 1e-5:
            print("  [PASS] Accumulation Accurate")
            return True
        else:
            print("  [FAIL] Accumulation Drift")
            return False

def main():
    cert = GPUPrecisionCertifier()
    p1 = cert.test_large_numbers()
    p2 = cert.test_accumulation()
    
    results = {
        "large_number_precision": "PASS" if p1 else "FAIL",
        "accumulation_precision": "PASS" if p2 else "FAIL"
    }
    
    with open("gpu_precision_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
