"""
Quick validation script for HNS accumulative test fix
======================================================
Runs only the critical accumulative test to validate the fix.
"""

import json
import time
from decimal import Decimal, getcontext
import math

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

def float_to_hns(value: float, precision: int = 6):
    """Converts a float to HNS with precision scaling."""
    if value == 0.0:
        return [0.0, 0.0, 0.0, 0.0]

    # Scale to integer
    scaled_value = int(round(value * (10 ** precision)))

    # Convert integer to HNS hierarchical levels
    r = float(scaled_value % int(BASE))
    scaled_value //= int(BASE)
    g = float(scaled_value % int(BASE))
    scaled_value //= int(BASE)
    b = float(scaled_value % int(BASE))
    scaled_value //= int(BASE)
    a = float(scaled_value)

    return [r, g, b, a]

def hns_to_decimal(vec4, precision=6):
    """Converts HNS to Decimal with precision unscaling."""
    r, g, b, a = vec4
    integer_value = Decimal(r) + Decimal(g) * Decimal(BASE) + Decimal(b) * Decimal(BASE)**2 + Decimal(a) * Decimal(BASE)**3
    return integer_value / Decimal(10 ** precision)

def hns_add(vec_a, vec_b):
    """HNS hierarchical addition."""
    raw_sum = [x + y for x, y in zip(vec_a, vec_b)]
    return hns_normalize(raw_sum)

print("="*80)
print("HNS ACCUMULATIVE TEST VALIDATION")
print("="*80)
print("\nTesting the fix for Issue-001 (P0 Critical)")
print("-"*80)

iterations = 1000000
increment = 0.000001  # 1 micro
precision = 6  # 6 decimal places for microsecond precision
expected = iterations * increment  # Should be 1.0

print(f"\nTest Configuration:")
print(f"  Iterations: {iterations:,}")
print(f"  Increment:  {increment} (1 microsecond)")
print(f"  Precision:  {precision} decimal places")
print(f"  Expected:   {expected}")

# Standard float (reference)
print(f"\nRunning float accumulation...")
float_value = 0.0
start_time = time.perf_counter()
for _ in range(iterations):
    float_value += increment
float_time = time.perf_counter() - start_time
float_error = abs(float_value - expected)

print(f"  Result: {float_value:.15f}")
print(f"  Error:  {float_error:.2e}")
print(f"  Time:   {float_time:.4f}s")

# HNS (with precision scaling for small floats)
print(f"\nRunning HNS accumulation (with precision scaling)...")
hns_value = [0.0, 0.0, 0.0, 0.0]
hns_increment = float_to_hns(increment, precision=precision)

print(f"  HNS increment: {hns_increment}")

start_time = time.perf_counter()
for i in range(iterations):
    hns_value = hns_add(hns_value, hns_increment)
    if i % 200000 == 0 and i > 0:
        current = hns_to_decimal(hns_value, precision=precision)
        print(f"    Progress {i:>9,}: {float(current):.6f}")

hns_time = time.perf_counter() - start_time
hns_result = hns_to_decimal(hns_value, precision=precision)
hns_error = abs(float(hns_result) - expected)

print(f"  Final HNS state: {hns_value}")
print(f"  Result: {float(hns_result):.15f}")
print(f"  Error:  {hns_error:.2e}")
print(f"  Time:   {hns_time:.4f}s")

# Decimal (reference)
print(f"\nRunning Decimal accumulation...")
dec_value = Decimal(0)
dec_increment = Decimal(increment)
start_time = time.perf_counter()
for _ in range(iterations):
    dec_value += dec_increment
dec_time = time.perf_counter() - start_time
dec_error = abs(float(dec_value) - expected)

print(f"  Result: {float(dec_value):.15f}")
print(f"  Error:  {dec_error:.2e}")
print(f"  Time:   {dec_time:.4f}s")

# Validation
print("\n" + "="*80)
print("VALIDATION RESULTS")
print("="*80)

results = {
    "test": "HNS Accumulative Precision Test (Issue-001)",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "configuration": {
        "iterations": iterations,
        "increment": increment,
        "precision": precision,
        "expected": expected
    },
    "results": {
        "float": {
            "result": float_value,
            "error": float_error,
            "time": float_time,
            "status": "PASS" if float_error < 1e-6 else "FAIL"
        },
        "hns": {
            "result": float(hns_result),
            "error": hns_error,
            "time": hns_time,
            "overhead": hns_time / float_time,
            "status": "PASS" if hns_error < 1e-9 else "FAIL"
        },
        "decimal": {
            "result": float(dec_value),
            "error": dec_error,
            "time": dec_time,
            "status": "PASS" if dec_error < 1e-9 else "FAIL"
        }
    }
}

# Print summary
print(f"\nFloat:   {results['results']['float']['status']}")
print(f"  Error: {float_error:.2e}")
print(f"\nHNS:     {results['results']['hns']['status']}")
print(f"  Error: {hns_error:.2e} (target: < 1e-09)")
print(f"  Overhead: {results['results']['hns']['overhead']:.2f}x slower than float")
print(f"\nDecimal: {results['results']['decimal']['status']}")
print(f"  Error: {dec_error:.2e}")

# Final verdict
print("\n" + "="*80)
if hns_error < 1e-9:
    print("✓✓✓ SUCCESS: HNS ACCUMULATIVE TEST PASSED ✓✓✓")
    print("Issue-001 (P0 Critical) is FIXED")
    print("\nHNS achieves perfect precision (< 1e-09 error)")
    print("Ready for Phase 5 (Scientific Validation)")
else:
    print("✗✗✗ FAILURE: HNS ACCUMULATIVE TEST FAILED ✗✗✗")
    print("Issue-001 (P0 Critical) still present")
    print(f"Error: {hns_error:.2e} (expected < 1e-09)")

print("="*80)

# Save results to JSON
output_file = "hns_accumulative_validation_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("\nValidation complete.")
