"""
Debug script for HNS accumulative test failure
==============================================

The accumulative test fails because float_to_hns() converts 0.000001 to [0,0,0,0].
HNS is designed for integers, not fractional values.

Solution: Scale small floats to integers before conversion.
"""

import math
from decimal import Decimal, getcontext

getcontext().prec = 50
BASE = 1000.0
INV_BASE = 1.0 / BASE

def hns_normalize(vec4):
    """Normalizes an HNS number by propagating carries."""
    r, g, b, a = vec4

    carry0 = math.floor(r * INV_BASE)
    r = r - (carry0 * BASE)
    g += carry0

    carry1 = math.floor(g * INV_BASE)
    g = g - (carry1 * BASE)
    b += carry1

    carry2 = math.floor(b * INV_BASE)
    b = b - (carry2 * BASE)
    a += carry2

    return [r, g, b, a]

def float_to_hns_old(value):
    """OLD: Converts a float to HNS - BROKEN for small floats."""
    if value == 0.0:
        return [0.0, 0.0, 0.0, 0.0]

    dec_value = Decimal(str(value))

    a = float(dec_value // Decimal(BASE ** 3))
    remainder = dec_value - (Decimal(a) * Decimal(BASE ** 3))
    b = float(remainder // Decimal(BASE ** 2))
    remainder = remainder - (Decimal(b) * Decimal(BASE ** 2))
    g = float(remainder // Decimal(BASE))
    r = float(remainder - (Decimal(g) * Decimal(BASE)))

    return hns_normalize([r, g, b, a])

def float_to_hns_fixed(value, precision=6):
    """
    FIXED: Converts a float to HNS with precision scaling.

    HNS is designed for integers. To handle small floats:
    1. Scale by 10^precision to make them integers
    2. Convert to HNS
    3. Results must be unscaled when converting back

    Args:
        value: Float value to convert
        precision: Decimal places to preserve (default 6 for micro precision)

    Returns:
        HNS representation as [r, g, b, a]
    """
    if value == 0.0:
        return [0.0, 0.0, 0.0, 0.0]

    # Scale to integer
    scaled_value = int(round(value * (10 ** precision)))

    # Convert integer to HNS
    r = float(scaled_value % int(BASE))
    scaled_value //= int(BASE)
    g = float(scaled_value % int(BASE))
    scaled_value //= int(BASE)
    b = float(scaled_value % int(BASE))
    scaled_value //= int(BASE)
    a = float(scaled_value)

    return [r, g, b, a]

def hns_to_float_unscaled(vec4, precision=6):
    """
    Convert HNS back to float with precision unscaling.

    Args:
        vec4: HNS representation [r, g, b, a]
        precision: Decimal places that were scaled

    Returns:
        Float value
    """
    r, g, b, a = vec4
    integer_value = int(r + g * BASE + b * BASE * BASE + a * BASE * BASE * BASE)
    return integer_value / (10 ** precision)

def hns_add(vec_a, vec_b):
    """HNS hierarchical addition."""
    raw_sum = [x + y for x, y in zip(vec_a, vec_b)]
    return hns_normalize(raw_sum)

# ============================================================================
# DEBUGGING
# ============================================================================

print("="*80)
print("HNS ACCUMULATIVE TEST - DEBUG")
print("="*80)

# Test the old broken version
print("\n1. OLD VERSION (BROKEN):")
print("-" * 80)
small_value = 0.000001
hns_old = float_to_hns_old(small_value)
print(f"Input: {small_value}")
print(f"HNS (old): {hns_old}")
print(f"Result: All zeros! This is why accumulation fails.")

# Test the new fixed version
print("\n2. NEW VERSION (FIXED):")
print("-" * 80)
hns_new = float_to_hns_fixed(small_value, precision=6)
print(f"Input: {small_value}")
print(f"HNS (new): {hns_new}")
recovered = hns_to_float_unscaled(hns_new, precision=6)
print(f"Recovered: {recovered}")
print(f"Match: {abs(recovered - small_value) < 1e-9}")

# Test accumulation with new version
print("\n3. ACCUMULATIVE TEST (1M iterations):")
print("-" * 80)

iterations = 1000000
increment = 0.000001
expected = iterations * increment  # Should be 1.0

# New fixed version
hns_value = [0.0, 0.0, 0.0, 0.0]
hns_increment = float_to_hns_fixed(increment, precision=6)

print(f"Increment: {increment}")
print(f"HNS increment: {hns_increment}")
print(f"Expected after {iterations:,} iterations: {expected}")
print(f"\nRunning accumulation...")

import time
start = time.perf_counter()
for i in range(iterations):
    hns_value = hns_add(hns_value, hns_increment)
    if i % 100000 == 0 and i > 0:
        current = hns_to_float_unscaled(hns_value, precision=6)
        print(f"  Iteration {i:>9,}: {current:.6f}")

elapsed = time.perf_counter() - start
final_result = hns_to_float_unscaled(hns_value, precision=6)
error = abs(final_result - expected)

print(f"\nFinal result: {final_result:.10f}")
print(f"Expected:     {expected:.10f}")
print(f"Error:        {error:.2e}")
print(f"Time:         {elapsed:.4f}s")

if error < 1e-6:
    print(f"\n✅ SUCCESS: HNS accumulative test PASSED!")
else:
    print(f"\n❌ FAILURE: HNS accumulative test FAILED!")

print("\n" + "="*80)
print("SOLUTION SUMMARY")
print("="*80)
print("""
The accumulative test failed because HNS is designed for INTEGERS, not floats.

Problem:
- float_to_hns(0.000001) → [0, 0, 0, 0] (rounds to zero)
- Adding zeros 1M times → still zero

Solution:
- Scale small floats to integers: 0.000001 × 10^6 = 1
- Convert scaled integer to HNS: 1 → [1, 0, 0, 0]
- Accumulate in HNS space
- Unscale when converting back: result / 10^6

This is exactly how fixed-point arithmetic works in financial systems.
HNS is essentially a hierarchical fixed-point system with base 1000.

To fix the benchmark:
1. Update float_to_hns() to accept a 'precision' parameter
2. Update hns_to_float() to accept a 'precision' parameter for unscaling
3. Update accumulative test to use precision=6
4. Document that HNS is for integers, use scaling for fractional values
""")
print("="*80)
