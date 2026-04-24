// =============================================================================
// HNS CORE SHADER LIBRARY - HIERARCHICAL NUMBER SYSTEM
// Veselov/Angulo Implementation for GPU
// =============================================================================
// 
// This shader library implements the Hierarchical Number System for extended
// precision arithmetic on GPU textures. All operations preserve exact integer
// values up to 10^12 using 4 RGBA channels as hierarchical levels.
//
// Usage:
//   #include "hns_core.glsl"
//   
//   HNumber activation = texelFetch(u_state, coord, 0);
//   HNumber weighted = hns_scale(activation, synaptic_weight);
//   HNumber accumulated = hns_add(total, weighted);
//
// Authors: V.F. Veselov (MIET), Francisco Angulo de Lafuente (Madrid)
// =============================================================================

#ifndef HNS_CORE_GLSL
#define HNS_CORE_GLSL

// =============================================================================
// CONFIGURATION
// =============================================================================

// Hierarchical base value
// Using 1000 provides good balance between range and precision
// Each channel holds values 0-999, giving total range 0 to 10^12-1
const float HNS_BASE = 1000.0;
const float HNS_INV_BASE = 0.001;  // 1.0 / 1000.0 - precomputed for efficiency

// Precision thresholds
const float HNS_EPSILON = 0.0001;

// =============================================================================
// TYPE DEFINITION
// =============================================================================

// An HNumber is a vec4 where:
//   .r (x) = Level 0: Units (0-999)
//   .g (y) = Level 1: Thousands (0-999)
//   .b (z) = Level 2: Millions (0-999)
//   .a (w) = Level 3: Billions (0-999+, can exceed for overflow)
#define HNumber vec4

// =============================================================================
// CORE OPERATIONS
// =============================================================================

/**
 * Normalize an HNumber by propagating carries between levels.
 * 
 * This is the "Veselov secret" - the key innovation that enables parallel
 * addition. After adding two HNumbers channel-wise, some channels may
 * exceed BASE. This function propagates the overflow to higher levels.
 *
 * @param n Input HNumber (may have overflow in channels)
 * @return Normalized HNumber with channels in valid range [0, BASE)
 *
 * Performance: ~4 floor operations, highly parallel on GPU
 */
HNumber hns_normalize(HNumber n) {
    HNumber res = n;
    
    // Level 0 → Level 1 carry
    // Using floor(x * INV_BASE) instead of floor(x / BASE) for performance
    float carry0 = floor(res.r * HNS_INV_BASE);
    res.r = res.r - (carry0 * HNS_BASE);  // Equivalent to mod but faster/safer
    res.g += carry0;
    
    // Level 1 → Level 2 carry
    float carry1 = floor(res.g * HNS_INV_BASE);
    res.g = res.g - (carry1 * HNS_BASE);
    res.b += carry1;
    
    // Level 2 → Level 3 carry
    float carry2 = floor(res.b * HNS_INV_BASE);
    res.b = res.b - (carry2 * HNS_BASE);
    res.a += carry2;
    
    // Note: Level 3 (Alpha) can grow beyond BASE, up to float32 max (~3.4e38)
    // This is intentional for accumulating very large synaptic sums
    
    return res;
}

/**
 * Hierarchical addition of two HNumbers.
 *
 * Algorithm (Veselov Algorithm 1):
 * 1. Parallel SIMD add all 4 channels (1 GPU cycle)
 * 2. Normalize to propagate carries
 *
 * @param a First operand
 * @param b Second operand
 * @return Sum as normalized HNumber
 *
 * Example:
 *   a = vec4(999, 999, 0, 0)  // 999,999
 *   b = vec4(1, 0, 0, 0)       // 1
 *   result = hns_add(a, b)
 *   // → vec4(0, 0, 1, 0)       // 1,000,000
 */
HNumber hns_add(HNumber a, HNumber b) {
    // Step 1: Parallel addition (single SIMD operation on GPU)
    HNumber sum = a + b;
    
    // Step 2: Carry resolution
    return hns_normalize(sum);
}

/**
 * Add three HNumbers efficiently.
 * Useful for: input + recurrent + bias
 */
HNumber hns_add3(HNumber a, HNumber b, HNumber c) {
    HNumber sum = a + b + c;
    return hns_normalize(sum);
}

/**
 * Multiply HNumber by scalar value.
 * 
 * Primary use case: Synaptic weight multiplication
 *   weighted_input = hns_scale(presynaptic_activation, synaptic_weight)
 *
 * @param a HNumber to scale
 * @param scalar Multiplication factor (typically synaptic weight 0.0-1.0)
 * @return Scaled and normalized HNumber
 *
 * Note: For scalar > 1.0, values will grow; scalar < 1.0 causes decay
 */
HNumber hns_scale(HNumber a, float scalar) {
    // Multiply all levels by scalar
    HNumber scaled = a * scalar;
    
    // Normalize to handle fractional overflow
    return hns_normalize(scaled);
}

/**
 * Multiply HNumber by per-channel scalars.
 * Useful for: Different decay rates per hierarchical level
 */
HNumber hns_scale4(HNumber a, vec4 scalars) {
    HNumber scaled = a * scalars;
    return hns_normalize(scaled);
}

/**
 * Hierarchical subtraction: a - b
 * 
 * WARNING: Does not handle negative results properly.
 * Only use when a >= b is guaranteed.
 *
 * @param a Minuend (must be >= b)
 * @param b Subtrahend
 * @return Difference (undefined if a < b)
 */
HNumber hns_subtract(HNumber a, HNumber b) {
    // Add BASE to each level to prevent intermediate negatives
    // Then subtract 1 from each carry level to compensate
    HNumber diff;
    diff.r = a.r - b.r + HNS_BASE;
    diff.g = a.g - b.g + HNS_BASE - 1.0;
    diff.b = a.b - b.b + HNS_BASE - 1.0;
    diff.a = a.a - b.a - 1.0;
    
    return hns_normalize(diff);
}

/**
 * Linear interpolation between two HNumbers.
 * 
 * result = a * (1-t) + b * t
 *
 * @param a Start value
 * @param b End value
 * @param t Interpolation factor [0, 1]
 * @return Interpolated HNumber
 */
HNumber hns_mix(HNumber a, HNumber b, float t) {
    // Scale each by complement weights and add
    HNumber scaled_a = hns_scale(a, 1.0 - t);
    HNumber scaled_b = hns_scale(b, t);
    return hns_add(scaled_a, scaled_b);
}

// =============================================================================
// COMPARISON OPERATIONS
// =============================================================================

/**
 * Check if HNumber is zero.
 */
bool hns_is_zero(HNumber n) {
    return n.r < HNS_EPSILON && 
           n.g < HNS_EPSILON && 
           n.b < HNS_EPSILON && 
           n.a < HNS_EPSILON;
}

/**
 * Check if two HNumbers are approximately equal.
 */
bool hns_equal(HNumber a, HNumber b) {
    vec4 diff = abs(a - b);
    return diff.r < HNS_EPSILON && 
           diff.g < HNS_EPSILON && 
           diff.b < HNS_EPSILON && 
           diff.a < HNS_EPSILON;
}

/**
 * Compare two HNumbers: returns sign of (a - b)
 *  1.0 if a > b
 *  0.0 if a == b
 * -1.0 if a < b
 */
float hns_compare(HNumber a, HNumber b) {
    // Compare from most significant level
    if (a.a > b.a + HNS_EPSILON) return 1.0;
    if (a.a < b.a - HNS_EPSILON) return -1.0;
    
    if (a.b > b.b + HNS_EPSILON) return 1.0;
    if (a.b < b.b - HNS_EPSILON) return -1.0;
    
    if (a.g > b.g + HNS_EPSILON) return 1.0;
    if (a.g < b.g - HNS_EPSILON) return -1.0;
    
    if (a.r > b.r + HNS_EPSILON) return 1.0;
    if (a.r < b.r - HNS_EPSILON) return -1.0;
    
    return 0.0;
}

/**
 * Return maximum of two HNumbers.
 */
HNumber hns_max(HNumber a, HNumber b) {
    return (hns_compare(a, b) >= 0.0) ? a : b;
}

/**
 * Return minimum of two HNumbers.
 */
HNumber hns_min(HNumber a, HNumber b) {
    return (hns_compare(a, b) <= 0.0) ? a : b;
}

/**
 * Clamp HNumber to range [min_val, max_val].
 */
HNumber hns_clamp(HNumber n, HNumber min_val, HNumber max_val) {
    return hns_max(min_val, hns_min(n, max_val));
}

// =============================================================================
// CONVERSION OPERATIONS
// =============================================================================

/**
 * Convert HNumber to regular float (with potential precision loss).
 * Useful for: Final output, visualization, activation functions
 */
float hns_to_float(HNumber n) {
    return n.r + 
           (n.g * HNS_BASE) + 
           (n.b * HNS_BASE * HNS_BASE) + 
           (n.a * HNS_BASE * HNS_BASE * HNS_BASE);
}

/**
 * Convert float to HNumber.
 * Input is treated as integer value.
 */
HNumber hns_from_float(float value) {
    HNumber result = HNumber(0.0);
    
    result.r = mod(value, HNS_BASE);
    value = floor(value * HNS_INV_BASE);
    
    result.g = mod(value, HNS_BASE);
    value = floor(value * HNS_INV_BASE);
    
    result.b = mod(value, HNS_BASE);
    value = floor(value * HNS_INV_BASE);
    
    result.a = value;
    
    return result;
}

/**
 * Convert normalized activation (0-1) to HNumber with specified range.
 * Useful for: Converting standard neural activations to HNS
 *
 * @param activation Normalized value [0, 1]
 * @param max_value Maximum representable value
 * @return HNumber representing activation * max_value
 */
HNumber hns_from_activation(float activation, float max_value) {
    return hns_from_float(activation * max_value);
}

/**
 * Convert HNumber to normalized activation (0-1).
 *
 * @param n HNumber to convert
 * @param max_value Value that maps to 1.0
 * @return Normalized activation [0, 1]
 */
float hns_to_activation(HNumber n, float max_value) {
    return clamp(hns_to_float(n) / max_value, 0.0, 1.0);
}

// =============================================================================
// NEURAL NETWORK OPERATIONS
// =============================================================================

/**
 * Synaptic accumulation: accumulate weighted input.
 * Core operation for neuromorphic computing.
 *
 * @param accumulator Current accumulated value
 * @param input Presynaptic activation
 * @param weight Synaptic weight
 * @return New accumulated value
 */
HNumber hns_synapse(HNumber accumulator, HNumber input, float weight) {
    HNumber weighted = hns_scale(input, weight);
    return hns_add(accumulator, weighted);
}

/**
 * Leaky integration with decay.
 * Models biological neuron membrane dynamics.
 *
 * dV/dt = -V/tau + I
 *
 * @param state Current membrane potential (HNumber)
 * @param input Synaptic input (HNumber)
 * @param decay Decay factor (1 - dt/tau), typically 0.9-0.99
 * @return Updated state
 */
HNumber hns_leaky_integrate(HNumber state, HNumber input, float decay) {
    HNumber decayed = hns_scale(state, decay);
    return hns_add(decayed, input);
}

/**
 * Soft threshold activation.
 * Converts accumulated HNumber to firing probability.
 *
 * Uses sigmoid-like curve on the total value.
 *
 * @param n Accumulated input (HNumber)
 * @param threshold Firing threshold
 * @param steepness Sigmoid steepness
 * @return Firing probability [0, 1]
 */
float hns_soft_threshold(HNumber n, float threshold, float steepness) {
    float total = hns_to_float(n);
    return 1.0 / (1.0 + exp(-steepness * (total - threshold)));
}

/**
 * Hard threshold activation.
 * Binary firing decision.
 *
 * @param n Accumulated input (HNumber)
 * @param threshold Firing threshold
 * @return 1.0 if above threshold, 0.0 otherwise
 */
float hns_hard_threshold(HNumber n, float threshold) {
    return (hns_to_float(n) > threshold) ? 1.0 : 0.0;
}

// =============================================================================
// NEIGHBORHOOD OPERATIONS (for Cellular Automata)
// =============================================================================

/**
 * Weighted sum of 3x3 neighborhood using HNS.
 * Assumes textures store HNumbers in RGBA.
 *
 * @param tex Source texture
 * @param coord Center coordinate
 * @param kernel 3x3 weight kernel (as array of 9 floats)
 * @param texel_size 1.0 / texture_size
 * @return Weighted sum as HNumber
 */
HNumber hns_convolve_3x3(sampler2D tex, ivec2 coord, float kernel[9], vec2 texel_size) {
    HNumber sum = HNumber(0.0);
    int idx = 0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 offset = ivec2(dx, dy);
            HNumber neighbor = texelFetch(tex, coord + offset, 0);
            sum = hns_synapse(sum, neighbor, kernel[idx]);
            idx++;
        }
    }
    
    return sum;
}

/**
 * Accumulate same-colored neighbors (for density calculation).
 *
 * @param tex Source texture
 * @param coord Center coordinate
 * @param center_value Center HNumber value
 * @param tolerance Matching tolerance
 * @return Count of matching neighbors as HNumber
 */
HNumber hns_count_similar_neighbors(sampler2D tex, ivec2 coord, HNumber center_value, float tolerance) {
    HNumber count = HNumber(0.0);
    HNumber one = HNumber(1.0, 0.0, 0.0, 0.0);
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;  // Skip center
            
            ivec2 offset = ivec2(dx, dy);
            HNumber neighbor = texelFetch(tex, coord + offset, 0);
            
            // Check if similar (using float conversion for comparison)
            float diff = abs(hns_to_float(neighbor) - hns_to_float(center_value));
            if (diff < tolerance) {
                count = hns_add(count, one);
            }
        }
    }
    
    return count;
}

// =============================================================================
// HEBBIAN PLASTICITY
// =============================================================================

/**
 * Hebbian weight update with homeostatic regulation.
 * 
 * Δw = η(x_i * x_j - <x_i><x_j>) + λ(1 - w²)
 *
 * @param weight Current weight (as float)
 * @param pre Presynaptic activation (HNumber)
 * @param post Postsynaptic activation (HNumber)
 * @param mean_pre Mean presynaptic activation
 * @param mean_post Mean postsynaptic activation
 * @param learning_rate η
 * @param regularization λ (homeostatic term)
 * @return Updated weight
 */
float hns_hebbian_update(
    float weight,
    HNumber pre,
    HNumber post,
    float mean_pre,
    float mean_post,
    float learning_rate,
    float regularization
) {
    // Convert to float for correlation computation
    float pre_f = hns_to_float(pre);
    float post_f = hns_to_float(post);
    
    // Hebbian term: correlation minus baseline
    float hebbian = pre_f * post_f - mean_pre * mean_post;
    
    // Homeostatic regularization: prevents weights from growing unbounded
    float homeostatic = 1.0 - weight * weight;
    
    // Update
    float delta = learning_rate * (hebbian + regularization * homeostatic);
    
    return clamp(weight + delta, -1.0, 1.0);
}

// =============================================================================
// HOLOGRAPHIC MEMORY OPERATIONS
// =============================================================================

/**
 * Encode association into holographic memory.
 * M ← M + α · φ(input) ⊗ φ(output)^T
 *
 * Simplified for single-pixel update.
 *
 * @param memory Current memory state
 * @param input Input pattern (projected)
 * @param output Output pattern (projected)
 * @param learning_rate α
 * @return Updated memory state
 */
HNumber hns_holo_encode(HNumber memory, HNumber input, HNumber output, float learning_rate) {
    // Outer product approximation for single element
    float interference = hns_to_float(input) * hns_to_float(output);
    HNumber encoded = hns_from_float(interference * learning_rate);
    return hns_add(memory, encoded);
}

/**
 * Retrieve from holographic memory.
 * R = M ⊙ φ(query)
 *
 * @param memory Memory state
 * @param query Query pattern
 * @return Retrieved association
 */
HNumber hns_holo_retrieve(HNumber memory, HNumber query) {
    float query_val = hns_to_float(query);
    return hns_scale(memory, query_val);
}

// =============================================================================
// UTILITY MACROS
// =============================================================================

// Zero HNumber constant
#define HNS_ZERO HNumber(0.0, 0.0, 0.0, 0.0)

// Unit HNumber (value = 1)
#define HNS_ONE HNumber(1.0, 0.0, 0.0, 0.0)

// Maximum representable value (999,999,999,999)
#define HNS_MAX HNumber(999.0, 999.0, 999.0, 999.0)

#endif // HNS_CORE_GLSL
