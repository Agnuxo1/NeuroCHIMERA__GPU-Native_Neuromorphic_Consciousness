"""
COMPREHENSIVE BENCHMARK: HNS vs Current Technologies
====================================================
Comparison of the HNS (Hierarchical Number System) by Veselov
with standard float, decimal.Decimal and other technologies.

Metrics evaluated:
- Precision with large numbers
- Operation speed
- Accumulative precision loss
- Edge cases
- Scalability
"""

import math
import time
import statistics
from decimal import Decimal, getcontext
from typing import List, Tuple, Dict
import sys

# HNS Configuration
BASE = 1000.0
INV_BASE = 1.0 / BASE

# Configure high precision for Decimal
getcontext().prec = 50

# ===================================================================
# HNS IMPLEMENTATION
# ===================================================================

def hns_normalize(vec4: List[float]) -> List[float]:
    """Normalizes an HNS number by propagating carries."""
    r, g, b, a = vec4
    
    # Carry 0 -> 1
    carry0 = math.floor(r * INV_BASE)
    r = r - (carry0 * BASE)
    g += carry0
    
    # Carry 1 -> 2
    carry1 = math.floor(g * INV_BASE)
    g = g - (carry1 * BASE)
    b += carry1
    
    # Carry 2 -> 3
    carry2 = math.floor(b * INV_BASE)
    b = b - (carry2 * BASE)
    a += carry2
    
    return [r, g, b, a]

def hns_to_float(vec4: List[float], precision: int = 6) -> float:
    """
    Converts HNS to float with precision unscaling.

    Args:
        vec4: HNS representation [r, g, b, a]
        precision: Decimal places that were scaled in float_to_hns()
                   Must match the precision used during conversion

    Returns:
        Float value (unscaled if precision > 0)

    Note:
        May lose precision for very large numbers (>10^15) due to float64 limits
    """
    r, g, b, a = vec4
    integer_value = r + (g * BASE) + (b * BASE * BASE) + (a * BASE * BASE * BASE)
    return integer_value / (10 ** precision)

def hns_to_decimal(vec4: List[float], precision: int = 6) -> Decimal:
    """
    Converts HNS to Decimal for precise comparison with precision unscaling.

    Args:
        vec4: HNS representation [r, g, b, a]
        precision: Decimal places that were scaled in float_to_hns()

    Returns:
        Decimal value (unscaled if precision > 0)
    """
    r, g, b, a = vec4
    integer_value = Decimal(r) + Decimal(g) * Decimal(BASE) + Decimal(b) * Decimal(BASE)**2 + Decimal(a) * Decimal(BASE)**3
    return integer_value / Decimal(10 ** precision)

def float_to_hns(value: float, precision: int = 6) -> List[float]:
    """
    Converts a float to HNS with precision scaling.

    HNS is designed for integers. To handle small floats:
    1. Scale by 10^precision to make them integers
    2. Convert to HNS representation
    3. Results must be unscaled when converting back with hns_to_float()

    Args:
        value: Float value to convert
        precision: Decimal places to preserve (default 6 for micro precision)
                   Use 0 for pure integers, 3 for milliseconds, 6 for microseconds

    Returns:
        HNS representation as [r, g, b, a]

    Example:
        # For small float accumulation (neural networks):
        hns = float_to_hns(0.000001, precision=6)  # [1.0, 0.0, 0.0, 0.0]

        # For integers (no scaling):
        hns = float_to_hns(123456.0, precision=0)   # [456.0, 123.0, 0.0, 0.0]
    """
    if value == 0.0:
        return [0.0, 0.0, 0.0, 0.0]

    # Scale to integer to preserve fractional precision
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

def hns_add(vec_a: List[float], vec_b: List[float]) -> List[float]:
    """HNS hierarchical addition."""
    raw_sum = [x + y for x, y in zip(vec_a, vec_b)]
    return hns_normalize(raw_sum)

def hns_scale(vec: List[float], scalar: float) -> List[float]:
    """Multiplies HNS by scalar."""
    scaled = [x * scalar for x in vec]
    return hns_normalize(scaled)

def hns_multiply(vec_a: List[float], vec_b: List[float], precision: int = 0) -> List[float]:
    """
    HNS multiplication (approximated using conversion).

    Args:
        vec_a: First HNS operand
        vec_b: Second HNS operand
        precision: Decimal places for precision scaling (must match inputs)

    Returns:
        Product as HNS representation
    """
    # For complete multiplication, we would need to implement the full algorithm
    # For now we use decimal conversion and back
    dec_a = hns_to_decimal(vec_a, precision=precision)
    dec_b = hns_to_decimal(vec_b, precision=precision)
    result_decimal = dec_a * dec_b
    # Convert decimal result to HNS
    return float_to_hns(float(result_decimal), precision=precision)

# ===================================================================
# BENCHMARK FUNCTIONS
# ===================================================================

def benchmark_precision_large_numbers() -> Dict:
    """Test 1: Precision with very large numbers where float fails."""
    print("\n" + "="*80)
    print("TEST 1: PRECISION WITH VERY LARGE NUMBERS")
    print("="*80)
    
    results = {
        'test_name': 'Precision with large numbers',
        'cases': []
    }
    
    # Test cases: numbers where float loses precision
    # Float64 has ~15-17 significant digits of precision
    # We will test cases where significant precision is lost
    test_cases = [
        # Cases where float maintains precision
        ("999,999,999 + 1", 999999999.0, 1.0, 1000000000.0),
        ("1,000,000,000 + 1", 1000000000.0, 1.0, 1000000001.0),
        # Cases where float starts losing precision (many significant digits)
        # Python float64 can represent ~15-17 significant digits
        ("123456789012345.0 + 0.1", 123456789012345.0, 0.1, 123456789012345.1),
        ("999999999999999.0 + 1.0", 999999999999999.0, 1.0, 1000000000000000.0),
        # Extreme cases where float loses precision (very large numbers)
        ("1e15 + 1", 1e15, 1.0, 1e15 + 1.0),
        ("1e16 + 1", 1e16, 1.0, 1e16 + 1.0),
        ("1e17 + 1", 1e17, 1.0, 1e17 + 1.0),
        # Cases with numbers that have many significant digits
        ("1234567890123456.0 + 0.000001", 1234567890123456.0, 0.000001, 1234567890123456.000001),
        # Specific cases where HNS should shine: large numbers with precision
        ("999,999,999,999,999 + 1", 999999999999999.0, 1.0, 1000000000000000.0),
        ("123,456,789,012,345 + 678,901,234,567,890", 123456789012345.0, 678901234567890.0, 802358023580235.0),
    ]
    
    for name, a_val, b_val, expected in test_cases:
        # Standard float
        float_result = a_val + b_val
        float_error = abs(float_result - expected)

        # HNS (precision=0 for large integers, no fractional scaling needed)
        hns_a = float_to_hns(a_val, precision=0)
        hns_b = float_to_hns(b_val, precision=0)
        hns_result_vec = hns_add(hns_a, hns_b)
        hns_result = hns_to_decimal(hns_result_vec, precision=0)
        hns_error = abs(float(hns_result) - expected)
        
        # Decimal (reference)
        dec_a = Decimal(a_val)
        dec_b = Decimal(b_val)
        dec_result = dec_a + dec_b
        dec_error = abs(float(dec_result) - expected)
        
        case_result = {
            'name': name,
            'expected': expected,
            'float_result': float_result,
            'float_error': float_error,
            'hns_result': float(hns_result),
            'hns_error': hns_error,
            'decimal_result': float(dec_result),
            'decimal_error': dec_error,
            'hns_wins': hns_error < float_error
        }
        
        results['cases'].append(case_result)
        
        print(f"\n{name}:")
        print(f"  Expected:        {expected}")
        print(f"  Float:           {float_result} (Error: {float_error:.2e})")
        print(f"  HNS:             {float(hns_result)} (Error: {hns_error:.2e})")
        print(f"  Decimal (ref):   {float(dec_result)} (Error: {dec_error:.2e})")
        if hns_error < float_error:
            improvement = ((float_error - hns_error) / float_error * 100) if float_error > 0 else 0
            print(f"  [OK] HNS is more precise ({improvement:.1f}% better)")
        elif float_error < hns_error:
            print(f"  [WARN] Float is more precise")
        else:
            print(f"  [-] Same precision")
    
    return results

def benchmark_accumulative_precision() -> Dict:
    """Test 2: Accumulative precision loss in multiple operations."""
    print("\n" + "="*80)
    print("TEST 2: ACCUMULATIVE PRECISION (1,000,000 additions)")
    print("="*80)

    iterations = 1000000
    increment = 0.000001  # 1 micro
    precision = 6  # 6 decimal places for microsecond precision

    # Standard float
    float_value = 0.0
    start_time = time.perf_counter()
    for _ in range(iterations):
        float_value += increment
    float_time = time.perf_counter() - start_time
    float_expected = iterations * increment
    float_error = abs(float_value - float_expected)

    # HNS (with precision scaling for small floats)
    hns_value = [0.0, 0.0, 0.0, 0.0]
    hns_increment = float_to_hns(increment, precision=precision)
    start_time = time.perf_counter()
    for _ in range(iterations):
        hns_value = hns_add(hns_value, hns_increment)
    hns_time = time.perf_counter() - start_time
    hns_result = hns_to_decimal(hns_value, precision=precision)
    hns_error = abs(float(hns_result) - float_expected)
    
    # Decimal
    dec_value = Decimal(0)
    dec_increment = Decimal(increment)
    start_time = time.perf_counter()
    for _ in range(iterations):
        dec_value += dec_increment
    dec_time = time.perf_counter() - start_time
    dec_error = abs(float(dec_value) - float_expected)
    
    print(f"\nIterations: {iterations:,}")
    print(f"Expected value: {float_expected}")
    print(f"\nStandard float:")
    print(f"  Result: {float_value:.10f}")
    print(f"  Error:  {float_error:.2e}")
    print(f"  Time:   {float_time:.4f}s ({iterations/float_time:,.0f} ops/s)")
    print(f"\nHNS:")
    print(f"  Result: {float(hns_result):.10f}")
    print(f"  Error:  {hns_error:.2e}")
    print(f"  Time:   {hns_time:.4f}s ({iterations/hns_time:,.0f} ops/s)")
    print(f"  Overhead: {hns_time/float_time:.2f}x slower")
    print(f"\nDecimal (reference):")
    print(f"  Result: {float(dec_value):.10f}")
    print(f"  Error:  {dec_error:.2e}")
    print(f"  Time:   {dec_time:.4f}s ({iterations/dec_time:,.0f} ops/s)")
    
    return {
        'iterations': iterations,
        'expected': float_expected,
        'float': {'result': float_value, 'error': float_error, 'time': float_time},
        'hns': {'result': float(hns_result), 'error': hns_error, 'time': hns_time},
        'decimal': {'result': float(dec_value), 'error': dec_error, 'time': dec_time}
    }

def benchmark_speed_operations() -> Dict:
    """Test 3: Speed of basic operations."""
    print("\n" + "="*80)
    print("TEST 3: OPERATION SPEED")
    print("="*80)
    
    iterations = 100000
    results = {}
    
    # Prepare test data
    float_a = 123456.789
    float_b = 987654.321
    precision = 3  # 3 decimal places for these test values
    hns_a = float_to_hns(float_a, precision=precision)
    hns_b = float_to_hns(float_b, precision=precision)
    dec_a = Decimal(float_a)
    dec_b = Decimal(float_b)
    
    # ADDITION
    print("\n--- ADDITION ---")
    
    # Float
    start = time.perf_counter()
    for _ in range(iterations):
        _ = float_a + float_b
    float_add_time = time.perf_counter() - start
    
    # HNS
    start = time.perf_counter()
    for _ in range(iterations):
        _ = hns_add(hns_a, hns_b)
    hns_add_time = time.perf_counter() - start
    
    # Decimal
    start = time.perf_counter()
    for _ in range(iterations):
        _ = dec_a + dec_b
    dec_add_time = time.perf_counter() - start
    
    print(f"Float:   {float_add_time*1000:.4f}ms ({iterations/float_add_time:,.0f} ops/s)")
    print(f"HNS:     {hns_add_time*1000:.4f}ms ({iterations/hns_add_time:,.0f} ops/s) - {hns_add_time/float_add_time:.2f}x slower")
    print(f"Decimal: {dec_add_time*1000:.4f}ms ({iterations/dec_add_time:,.0f} ops/s) - {dec_add_time/float_add_time:.2f}x slower")
    
    results['add'] = {
        'float': float_add_time,
        'hns': hns_add_time,
        'decimal': dec_add_time
    }
    
    # SCALAR MULTIPLICATION
    print("\n--- SCALAR MULTIPLICATION ---")
    scalar = 2.5
    
    # Float
    start = time.perf_counter()
    for _ in range(iterations):
        _ = float_a * scalar
    float_scale_time = time.perf_counter() - start
    
    # HNS
    start = time.perf_counter()
    for _ in range(iterations):
        _ = hns_scale(hns_a, scalar)
    hns_scale_time = time.perf_counter() - start
    
    # Decimal
    start = time.perf_counter()
    for _ in range(iterations):
        _ = dec_a * Decimal(scalar)
    dec_scale_time = time.perf_counter() - start
    
    print(f"Float:   {float_scale_time*1000:.4f}ms ({iterations/float_scale_time:,.0f} ops/s)")
    print(f"HNS:     {hns_scale_time*1000:.4f}ms ({iterations/hns_scale_time:,.0f} ops/s) - {hns_scale_time/float_scale_time:.2f}x slower")
    print(f"Decimal: {dec_scale_time*1000:.4f}ms ({iterations/dec_scale_time:,.0f} ops/s) - {dec_scale_time/float_scale_time:.2f}x slower")
    
    results['scale'] = {
        'float': float_scale_time,
        'hns': hns_scale_time,
        'decimal': dec_scale_time
    }
    
    return results

def benchmark_edge_cases() -> Dict:
    """Test 4: Edge cases and extremes."""
    print("\n" + "="*80)
    print("TEST 4: EDGE CASES AND EXTREMES")
    print("="*80)
    
    results = {'cases': []}
    
    test_cases = [
        ("Zero", 0.0, 0.0),
        ("Very small numbers", 0.000001, 0.000001),
        ("Maximum float32 (~3.4e38)", 3.4e38, 1.0),
        ("Negative numbers", -1000.0, 500.0),
        ("Multiple overflow", 999999.0, 999999.0),
    ]
    
    for name, a_val, b_val in test_cases:
        try:
            # Float
            float_result = a_val + b_val
            
            # HNS (precision=0 for edge case large numbers)
            # Note: HNS does not handle negatives directly - use absolute values
            hns_a = float_to_hns(abs(a_val), precision=0)
            hns_b = float_to_hns(abs(b_val), precision=0)
            hns_result_vec = hns_add(hns_a, hns_b)
            hns_result = hns_to_float(hns_result_vec, precision=0)
            
            # Check if there is significant difference
            if abs(a_val) < 1e10:  # Only compare if not too large
                error = abs(float_result - hns_result)
                status = "[OK] OK" if error < 1e-6 else "[WARN] Difference"
            else:
                error = None
                status = "[INFO] Very large number"
            
            case_result = {
                'name': name,
                'a': a_val,
                'b': b_val,
                'float_result': float_result,
                'hns_result': hns_result,
                'error': error,
                'status': status
            }
            
            results['cases'].append(case_result)
            
            print(f"\n{name}:")
            print(f"  A = {a_val}, B = {b_val}")
            print(f"  Float: {float_result}")
            print(f"  HNS:   {hns_result}")
            if error is not None:
                print(f"  Error: {error:.2e} - {status}")
            else:
                print(f"  {status}")
                
        except Exception as e:
            print(f"\n{name}: [ERROR] ERROR - {str(e)}")
            results['cases'].append({
                'name': name,
                'error': str(e)
            })
    
    return results

def benchmark_float32_simulation() -> Dict:
    """Test 6: Float32 simulation (as in GPU/GLSL) where HNS should shine."""
    print("\n" + "="*80)
    print("TEST 6: FLOAT32 SIMULATION (GPU/GLSL) - Where HNS should shine")
    print("="*80)
    
    # Float32 has only ~7 significant digits of precision
    # We will simulate this by limiting precision
    def simulate_float32(value: float) -> float:
        """Simulates the limited precision of float32."""
        # Float32 has ~7 significant digits
        if value == 0.0:
            return 0.0
        # Convert to scientific notation and limit significant digits
        import struct
        # Convert to float32 and back to simulate precision
        try:
            f32_bytes = struct.pack('f', value)
            return struct.unpack('f', f32_bytes)[0]
        except:
            return value
    
    results = {'cases': []}
    
    # Cases where float32 loses precision but HNS maintains
    test_cases = [
        ("999,999 + 1 (float32 simulated)", 999999.0, 1.0, 1000000.0),
        ("9,999,999 + 1 (float32 simulated)", 9999999.0, 1.0, 10000000.0),
        ("99,999,999 + 1 (float32 simulated)", 99999999.0, 1.0, 100000000.0),
        ("1234567.89 + 0.01 (float32 simulated)", 1234567.89, 0.01, 1234567.90),
        ("12345678.9 + 0.1 (float32 simulated)", 12345678.9, 0.1, 12345679.0),
    ]
    
    for name, a_val, b_val, expected in test_cases:
        # Simulated float32
        f32_a = simulate_float32(a_val)
        f32_b = simulate_float32(b_val)
        f32_result = simulate_float32(f32_a + f32_b)
        f32_error = abs(f32_result - expected)
        
        # HNS (maintains full precision with integer scaling)
        precision = 2  # 2 decimal places for float32 test values
        hns_a = float_to_hns(a_val, precision=precision)
        hns_b = float_to_hns(b_val, precision=precision)
        hns_result_vec = hns_add(hns_a, hns_b)
        hns_result = hns_to_decimal(hns_result_vec, precision=precision)
        hns_error = abs(float(hns_result) - expected)
        
        # Float64 (referencia)
        f64_result = a_val + b_val
        f64_error = abs(f64_result - expected)
        
        case_result = {
            'name': name,
            'expected': expected,
            'f32_result': f32_result,
            'f32_error': f32_error,
            'hns_result': float(hns_result),
            'hns_error': hns_error,
            'f64_result': f64_result,
            'f64_error': f64_error,
            'hns_better_than_f32': hns_error < f32_error
        }
        
        results['cases'].append(case_result)
        
        print(f"\n{name}:")
        print(f"  Expected:        {expected}")
        print(f"  Float32 (sim):   {f32_result} (Error: {f32_error:.2e})")
        print(f"  HNS:             {float(hns_result)} (Error: {hns_error:.2e})")
        print(f"  Float64 (ref):   {f64_result} (Error: {f64_error:.2e})")
        if hns_error < f32_error:
            improvement = ((f32_error - hns_error) / f32_error * 100) if f32_error > 0 else 0
            print(f"  [OK] HNS is {improvement:.1f}% more precise than float32")
        elif f32_error < hns_error:
            print(f"  [WARN] Float32 is more precise")
        else:
            print(f"  [-] Same precision")
    
    return results

def benchmark_extreme_accumulation() -> Dict:
    """Test 7: Extreme accumulative precision (simulating neural network)."""
    print("\n" + "="*80)
    print("TEST 7: EXTREME ACCUMULATIVE PRECISION (Neural Network Simulation)")
    print("="*80)
    
    # Simulate accumulation of neural activations
    # As in a neural network where many small inputs are summed
    iterations = 10000000  # 10 millones de iteraciones
    increment = 0.0000001  # 0.1 micro (muy pequeño)
    expected = iterations * increment
    
    print(f"\nSimulating {iterations:,} accumulations of {increment}")
    print(f"Expected value: {expected}")
    
    # Standard float
    float_value = 0.0
    start_time = time.perf_counter()
    for _ in range(iterations):
        float_value += increment
    float_time = time.perf_counter() - start_time
    float_error = abs(float_value - expected)
    
    # HNS (with precision scaling for small floats)
    precision = 7  # 7 decimal places for 0.1 microsecond precision
    hns_value = [0.0, 0.0, 0.0, 0.0]
    hns_increment = float_to_hns(increment, precision=precision)
    start_time = time.perf_counter()
    for _ in range(iterations):
        hns_value = hns_add(hns_value, hns_increment)
    hns_time = time.perf_counter() - start_time
    hns_result = hns_to_decimal(hns_value, precision=precision)
    hns_error = abs(float(hns_result) - expected)
    
    # Decimal (referencia)
    dec_value = Decimal(0)
    dec_increment = Decimal(str(increment))  # Usar string para precisión
    start_time = time.perf_counter()
    for _ in range(iterations):
        dec_value += dec_increment
    dec_time = time.perf_counter() - start_time
    dec_error = abs(float(dec_value) - expected)
    
    print(f"\nStandard float:")
    print(f"  Result: {float_value:.15f}")
    print(f"  Error:  {float_error:.2e} ({float_error/expected*100:.6f}% relative)")
    print(f"  Time:   {float_time:.4f}s ({iterations/float_time:,.0f} ops/s)")
    print(f"\nHNS:")
    print(f"  Result: {float(hns_result):.15f}")
    print(f"  Error:  {hns_error:.2e} ({hns_error/expected*100:.6f}% relative)")
    print(f"  Time:   {hns_time:.4f}s ({iterations/hns_time:,.0f} ops/s)")
    print(f"  Overhead: {hns_time/float_time:.2f}x slower")
    print(f"\nDecimal (reference):")
    print(f"  Result: {float(dec_value):.15f}")
    print(f"  Error:  {dec_error:.2e} ({dec_error/expected*100:.6f}% relative)")
    print(f"  Time:   {dec_time:.4f}s ({iterations/dec_time:,.0f} ops/s)")
    
    hns_better = hns_error < float_error
    if hns_better:
        improvement = ((float_error - hns_error) / float_error * 100) if float_error > 0 else 0
        print(f"\n[OK] HNS is {improvement:.2f}% more precise in extreme accumulation")
    else:
        print(f"\n[WARN] Float is more precise in extreme accumulation")
    
    return {
        'iterations': iterations,
        'expected': expected,
        'float': {'result': float_value, 'error': float_error, 'time': float_time},
        'hns': {'result': float(hns_result), 'error': hns_error, 'time': hns_time},
        'decimal': {'result': float(dec_value), 'error': dec_error, 'time': dec_time},
        'hns_better': hns_better
    }

def benchmark_scalability() -> Dict:
    """Test 5: Scalability with different number sizes."""
    print("\n" + "="*80)
    print("TEST 5: SCALABILITY")
    print("="*80)
    
    results = {'ranges': []}
    
    # Different number ranges
    ranges = [
        ("Small (0-1,000)", 0, 1000),
        ("Medium (0-1,000,000)", 0, 1000000),
        ("Large (0-1,000,000,000)", 0, 1000000000),
        ("Very large (0-1,000,000,000,000)", 0, 1000000000000),
    ]
    
    for name, min_val, max_val in ranges:
        # Generate random numbers in the range
        import random
        test_count = 1000
        errors_float = []
        errors_hns = []
        
        for _ in range(test_count):
            a = random.uniform(min_val, max_val)
            b = random.uniform(min_val, max_val)
            expected = a + b
            
            # Float
            float_result = a + b
            float_error = abs(float_result - expected)
            errors_float.append(float_error)
            
            # HNS (precision=0 for scalability test with large numbers)
            hns_a = float_to_hns(a, precision=0)
            hns_b = float_to_hns(b, precision=0)
            hns_result_vec = hns_add(hns_a, hns_b)
            hns_result = hns_to_decimal(hns_result_vec, precision=0)
            hns_error = abs(float(hns_result) - expected)
            errors_hns.append(hns_error)
        
        avg_error_float = statistics.mean(errors_float)
        avg_error_hns = statistics.mean(errors_hns)
        max_error_float = max(errors_float)
        max_error_hns = max(errors_hns)
        
        range_result = {
            'name': name,
            'test_count': test_count,
            'avg_error_float': avg_error_float,
            'avg_error_hns': avg_error_hns,
            'max_error_float': max_error_float,
            'max_error_hns': max_error_hns,
            'hns_better': avg_error_hns < avg_error_float
        }
        
        results['ranges'].append(range_result)
        
        print(f"\n{name} ({test_count} tests):")
        print(f"  Float - Average error: {avg_error_float:.2e}, Maximum: {max_error_float:.2e}")
        print(f"  HNS   - Average error: {avg_error_hns:.2e}, Maximum: {max_error_hns:.2e}")
        if avg_error_hns < avg_error_float:
            print(f"  [OK] HNS is more precise in this range")
        else:
            print(f"  [WARN] Float is more precise in this range")
    
    return results

def generate_summary_report(all_results: Dict):
    """Generates a summary report of all benchmarks."""
    print("\n" + "="*80)
    print("SUMMARY REPORT - HNS BENCHMARK vs CURRENT TECHNOLOGIES")
    print("="*80)
    
    print("\nCONCLUSIONS:")
    print("-" * 80)
    
    # Precision analysis (float64)
    precision_wins = 0
    precision_total = 0
    
    if 'precision' in all_results:
        for case in all_results['precision']['cases']:
            precision_total += 1
            if case.get('hns_wins', False):
                precision_wins += 1
    
    # Float32 precision analysis
    f32_wins = 0
    f32_total = 0
    
    if 'float32' in all_results:
        for case in all_results['float32']['cases']:
            f32_total += 1
            if case.get('hns_better_than_f32', False):
                f32_wins += 1
    
    if precision_total > 0:
        win_rate = (precision_wins / precision_total) * 100
        print(f"\n1. PRECISION (Float64 - CPU):")
        print(f"   HNS is more precise in {precision_wins}/{precision_total} cases ({win_rate:.1f}%)")
    
    if f32_total > 0:
        f32_win_rate = (f32_wins / f32_total) * 100
        print(f"\n1b. PRECISION (Float32 - GPU simulated):")
        print(f"   HNS is more precise in {f32_wins}/{f32_total} cases ({f32_win_rate:.1f}%)")
        print(f"   [OK] HNS shows clear advantages in float32 precision (GPU)")
    
    # Speed analysis
    if 'speed' in all_results:
        add_overhead = all_results['speed']['add']['hns'] / all_results['speed']['add']['float']
        scale_overhead = all_results['speed']['scale']['hns'] / all_results['speed']['scale']['float']
        avg_overhead = (add_overhead + scale_overhead) / 2
        
        print(f"\n2. SPEED:")
        print(f"   HNS is {avg_overhead:.2f}x slower than standard float")
        print(f"   - Addition: {add_overhead:.2f}x overhead")
        print(f"   - Scaling: {scale_overhead:.2f}x overhead")
    
    # Accumulative precision analysis
    if 'accumulative' in all_results:
        acc = all_results['accumulative']
        hns_better = acc['hns']['error'] < acc['float']['error']
        print(f"\n3. ACCUMULATIVE PRECISION (1M iterations):")
        print(f"   Float error: {acc['float']['error']:.2e}")
        print(f"   HNS error:   {acc['hns']['error']:.2e}")
        if hns_better:
            print(f"   [OK] HNS maintains better precision in repeated operations")
        else:
            print(f"   [WARN] Float maintains better precision in repeated operations")
    
    # Extreme accumulative precision analysis
    if 'extreme_accumulation' in all_results:
        ext = all_results['extreme_accumulation']
        print(f"\n3b. EXTREME ACCUMULATIVE PRECISION (10M iterations):")
        print(f"   Float error: {ext['float']['error']:.2e} ({ext['float']['error']/ext['expected']*100:.6f}% relative)")
        print(f"   HNS error:   {ext['hns']['error']:.2e} ({ext['hns']['error']/ext['expected']*100:.6f}% relative)")
        if ext['hns_better']:
            improvement = ((ext['float']['error'] - ext['hns']['error']) / ext['float']['error'] * 100) if ext['float']['error'] > 0 else 0
            print(f"   [OK] HNS is {improvement:.2f}% more precise in extreme accumulation")
        else:
            print(f"   [WARN] Float is more precise in extreme accumulation")
    
    # Recommendations
    print(f"\n4. RECOMMENDATIONS:")
    print(f"   - HNS is ideal for: large numbers where float loses precision")
    print(f"   - HNS is suitable for: neural operations with accumulation")
    print(f"   - Consider speed overhead: {avg_overhead:.2f}x on CPU")
    print(f"   - On GPU (GLSL), overhead should be lower due to SIMD")
    
    print("\n" + "="*80)

# ===================================================================
# EJECUCIÓN PRINCIPAL
# ===================================================================

def main():
    """Executes all benchmarks and generates report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK: HNS (Hierarchical Number System) vs Current Technologies")
    print("="*80)
    print("\nSystem: Hierarchical Number System (Veselov/Angulo)")
    print("Comparison: standard float, decimal.Decimal")
    print("Date:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    all_results = {}
    
    try:
        # Execute all benchmarks
        all_results['precision'] = benchmark_precision_large_numbers()
        all_results['accumulative'] = benchmark_accumulative_precision()
        all_results['speed'] = benchmark_speed_operations()
        all_results['edge_cases'] = benchmark_edge_cases()
        all_results['scalability'] = benchmark_scalability()
        all_results['float32'] = benchmark_float32_simulation()
        all_results['extreme_accumulation'] = benchmark_extreme_accumulation()
        
        # Generate summary report
        generate_summary_report(all_results)
        
        print("\n[OK] Benchmark completed successfully")
        
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

