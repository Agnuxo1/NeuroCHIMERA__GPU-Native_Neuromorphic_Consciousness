"""
Hierarchical Number System (HNS) - Veselov/Angulo Implementation
================================================================

This module implements the Hierarchical Number System for extended precision
arithmetic on GPU-compatible data structures. The HNS encodes large numbers
across 4 channels (RGBA) as hierarchical levels, enabling exact integer
arithmetic without floating-point precision loss.

Mathematical Foundation:
------------------------
A number N in HNS is represented as:
    N = R + G×BASE + B×BASE² + A×BASE³

Where BASE = 1000 (configurable), giving range:
    0 to 999,999,999,999 (10^12 - 1) with exact precision

Key Operations:
- Addition: Parallel SIMD add + cascading carry
- Scaling: Parallel multiply + carry propagation
- Normalization: The "Veselov secret" - ensures valid representation

Authors: V.F. Veselov (MIET), Francisco Angulo de Lafuente
"""

import math
from typing import List, Tuple, Union
import numpy as np


# Configuration - Base for hierarchical representation
BASE: float = 1000.0
INV_BASE: float = 0.001  # 1.0 / BASE for efficient division


class HNumber:
    """
    Hierarchical Number representation using 4 channels (vec4).
    
    Channels:
        R (index 0): Units level (0-999)
        G (index 1): Thousands level (0-999)
        B (index 2): Millions level (0-999)
        A (index 3): Billions level (0-999+)
    
    Example:
        1,234,567,890 → HNumber([890, 567, 234, 1])
    """
    
    __slots__ = ['_data']
    
    def __init__(self, data: Union[List[float], np.ndarray, 'HNumber', int, float] = None):
        """
        Initialize HNumber from various input types.
        
        Args:
            data: Can be:
                - List/array of 4 floats [R, G, B, A]
                - Another HNumber (copy)
                - Integer (auto-convert)
                - Float (auto-convert, rounded)
                - None (zero)
        """
        if data is None:
            self._data = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif isinstance(data, HNumber):
            self._data = data._data.copy()
        elif isinstance(data, (int, float)):
            self._data = self._from_number(data)
        elif isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        else:
            self._data = np.array(data, dtype=np.float32)
    
    @staticmethod
    def _from_number(value: Union[int, float]) -> np.ndarray:
        """Convert a regular number to HNS representation."""
        value = int(round(value))
        result = np.zeros(4, dtype=np.float32)
        
        for i in range(4):
            result[i] = value % int(BASE)
            value //= int(BASE)
        
        return result
    
    @property
    def r(self) -> float:
        """Units level (channel 0)."""
        return self._data[0]
    
    @r.setter
    def r(self, value: float):
        self._data[0] = value
    
    @property
    def g(self) -> float:
        """Thousands level (channel 1)."""
        return self._data[1]
    
    @g.setter
    def g(self, value: float):
        self._data[1] = value
    
    @property
    def b(self) -> float:
        """Millions level (channel 2)."""
        return self._data[2]
    
    @b.setter
    def b(self, value: float):
        self._data[2] = value
    
    @property
    def a(self) -> float:
        """Billions level (channel 3)."""
        return self._data[3]
    
    @a.setter
    def a(self, value: float):
        self._data[3] = value
    
    def __getitem__(self, index: int) -> float:
        return self._data[index]
    
    def __setitem__(self, index: int, value: float):
        self._data[index] = value
    
    def to_list(self) -> List[float]:
        """Return as Python list."""
        return self._data.tolist()
    
    def to_array(self) -> np.ndarray:
        """Return as numpy array."""
        return self._data.copy()
    
    def to_integer(self) -> int:
        """Convert back to standard integer."""
        return int(
            self._data[0] +
            self._data[1] * BASE +
            self._data[2] * BASE * BASE +
            self._data[3] * BASE * BASE * BASE
        )
    
    def to_float(self) -> float:
        """Convert back to standard float (may lose precision)."""
        return float(self.to_integer())
    
    def copy(self) -> 'HNumber':
        """Create a copy of this HNumber."""
        return HNumber(self._data.copy())
    
    def __repr__(self) -> str:
        return f"HNumber([{self.r:.0f}, {self.g:.0f}, {self.b:.0f}, {self.a:.0f}] = {self.to_integer():,})"
    
    def __str__(self) -> str:
        return f"[B:{int(self.a):3d} | M:{int(self.b):3d} | K:{int(self.g):3d} | U:{int(self.r):3d}]"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, HNumber):
            return np.allclose(self._data, other._data)
        elif isinstance(other, (list, np.ndarray)):
            return np.allclose(self._data, other)
        return False
    
    def __add__(self, other: 'HNumber') -> 'HNumber':
        return hns_add(self, other)
    
    def __mul__(self, scalar: float) -> 'HNumber':
        return hns_scale(self, scalar)
    
    def __rmul__(self, scalar: float) -> 'HNumber':
        return hns_scale(self, scalar)


def hns_normalize(n: HNumber) -> HNumber:
    """
    Normalize an HNumber by propagating carries between levels.
    
    This is the "Veselov secret" - enables parallel addition by
    deferring carry propagation to a separate pass.
    
    The algorithm:
    1. For each level L (0→3):
       - Compute carry = floor(L / BASE)
       - L = L - carry × BASE (equivalent to mod, but faster)
       - L+1 += carry
    
    Args:
        n: Input HNumber (may have overflow in channels)
    
    Returns:
        Normalized HNumber with all channels in [0, BASE)
    
    Note:
        The Alpha channel (level 3) can grow beyond BASE,
        up to float32 limit (~3.4e38). This is intentional
        for handling very large accumulated values.
    """
    result = n.copy()
    
    # Level 0 → Level 1 carry
    carry0 = math.floor(result.r * INV_BASE)
    result.r = result.r - (carry0 * BASE)
    result.g += carry0
    
    # Level 1 → Level 2 carry
    carry1 = math.floor(result.g * INV_BASE)
    result.g = result.g - (carry1 * BASE)
    result.b += carry1
    
    # Level 2 → Level 3 carry
    carry2 = math.floor(result.b * INV_BASE)
    result.b = result.b - (carry2 * BASE)
    result.a += carry2
    
    return result


def hns_add(a: HNumber, b: HNumber) -> HNumber:
    """
    Hierarchical addition of two HNumbers.
    
    Algorithm (Veselov Algorithm 1):
    1. Parallel SIMD addition of all 4 channels
    2. Normalize to propagate carries
    
    On GPU, step 1 executes in 1 clock cycle (vec4 + vec4).
    
    Args:
        a: First HNumber operand
        b: Second HNumber operand
    
    Returns:
        Sum as normalized HNumber
    
    Example:
        a = HNumber([999, 999, 0, 0])  # 999,999
        b = HNumber([1, 0, 0, 0])       # 1
        result = hns_add(a, b)
        # → HNumber([0, 0, 1, 0])       # 1,000,000
    """
    # Step 1: Parallel addition (vectorized)
    raw_sum = HNumber([
        a.r + b.r,
        a.g + b.g,
        a.b + b.b,
        a.a + b.a
    ])
    
    # Step 2: Carry resolution
    return hns_normalize(raw_sum)


def hns_subtract(a: HNumber, b: HNumber) -> HNumber:
    """
    Hierarchical subtraction: a - b.
    
    Note: Does not handle negative results properly.
    Only use when a >= b is guaranteed.
    
    Args:
        a: Minuend (must be >= b)
        b: Subtrahend
    
    Returns:
        Difference as normalized HNumber
    """
    # Borrow handling - add BASE to each level to prevent negatives
    result = HNumber([
        a.r - b.r + BASE,
        a.g - b.g + BASE - 1,  # Account for previous borrow
        a.b - b.b + BASE - 1,
        a.a - b.a - 1
    ])
    
    return hns_normalize(result)


def hns_scale(a: HNumber, scalar: float) -> HNumber:
    """
    Multiply HNumber by a scalar value (synaptic weight operation).
    
    Algorithm (Veselov Algorithm 2 simplified):
    1. Multiply all levels by scalar
    2. Normalize to redistribute overflow
    
    Args:
        a: HNumber to scale
        scalar: Multiplication factor (typically synaptic weight 0-1)
    
    Returns:
        Scaled and normalized HNumber
    
    Example:
        a = HNumber([500, 0, 0, 0])  # 500
        result = hns_scale(a, 2.5)
        # → HNumber([250, 1, 0, 0])   # 1,250
    """
    # Multiply all levels
    scaled = HNumber([
        a.r * scalar,
        a.g * scalar,
        a.b * scalar,
        a.a * scalar
    ])
    
    # Normalize to handle fractional carries
    return hns_normalize(scaled)


def hns_multiply(a: HNumber, b: HNumber) -> HNumber:
    """
    Full multiplication of two HNumbers.
    
    Uses schoolbook multiplication with carry propagation.
    More expensive than scale, but handles full precision.
    
    Args:
        a: First multiplicand
        b: Second multiplicand
    
    Returns:
        Product as normalized HNumber
    
    Note:
        Result may overflow float32 for very large inputs.
        Use for moderate values only.
    """
    # Initialize result with extended precision (8 levels conceptually)
    result = [0.0] * 8
    
    # Schoolbook multiplication
    for i in range(4):
        for j in range(4):
            if i + j < 8:
                result[i + j] += a[i] * b[j]
    
    # Normalize with extended carry
    for i in range(7):
        carry = math.floor(result[i] * INV_BASE)
        result[i] = result[i] - (carry * BASE)
        result[i + 1] += carry
    
    # Return truncated to 4 levels (may lose precision)
    return HNumber([result[0], result[1], result[2], result[3]])


def hns_compare(a: HNumber, b: HNumber) -> int:
    """
    Compare two HNumbers.
    
    Returns:
        -1 if a < b
         0 if a == b
         1 if a > b
    """
    # Compare from most significant level
    for i in range(3, -1, -1):
        if a[i] > b[i]:
            return 1
        elif a[i] < b[i]:
            return -1
    return 0


def hns_is_zero(n: HNumber) -> bool:
    """Check if HNumber is zero."""
    return n.r == 0 and n.g == 0 and n.b == 0 and n.a == 0


def hns_max(a: HNumber, b: HNumber) -> HNumber:
    """Return the larger of two HNumbers."""
    return a.copy() if hns_compare(a, b) >= 0 else b.copy()


def hns_min(a: HNumber, b: HNumber) -> HNumber:
    """Return the smaller of two HNumbers."""
    return a.copy() if hns_compare(a, b) <= 0 else b.copy()


def hns_clamp(n: HNumber, min_val: HNumber, max_val: HNumber) -> HNumber:
    """Clamp HNumber to range [min_val, max_val]."""
    return hns_max(min_val, hns_min(n, max_val))


# ============================================================================
# Batch Operations for Numpy Arrays
# ============================================================================

def hns_normalize_batch(data: np.ndarray) -> np.ndarray:
    """
    Normalize a batch of HNumbers stored as (N, 4) array.
    
    Vectorized for efficiency with large neuron populations.
    
    Args:
        data: Array of shape (N, 4) representing N HNumbers
    
    Returns:
        Normalized array of same shape
    """
    result = data.copy()
    
    # Level 0 → Level 1
    carry0 = np.floor(result[:, 0] * INV_BASE)
    result[:, 0] = result[:, 0] - (carry0 * BASE)
    result[:, 1] += carry0
    
    # Level 1 → Level 2
    carry1 = np.floor(result[:, 1] * INV_BASE)
    result[:, 1] = result[:, 1] - (carry1 * BASE)
    result[:, 2] += carry1
    
    # Level 2 → Level 3
    carry2 = np.floor(result[:, 2] * INV_BASE)
    result[:, 2] = result[:, 2] - (carry2 * BASE)
    result[:, 3] += carry2
    
    return result


def hns_add_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Add two batches of HNumbers.
    
    Args:
        a: Array of shape (N, 4)
        b: Array of shape (N, 4) or (4,) for broadcast
    
    Returns:
        Sum array of shape (N, 4)
    """
    raw_sum = a + b
    return hns_normalize_batch(raw_sum)


def hns_scale_batch(data: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    """
    Scale batch of HNumbers by individual scalars.
    
    Args:
        data: Array of shape (N, 4)
        scalars: Array of shape (N,) or (N, 1)
    
    Returns:
        Scaled array of shape (N, 4)
    """
    if scalars.ndim == 1:
        scalars = scalars[:, np.newaxis]
    
    scaled = data * scalars
    return hns_normalize_batch(scaled)


# ============================================================================
# Conversion Utilities
# ============================================================================

def integer_to_hns(value: int) -> HNumber:
    """Convert integer to HNumber."""
    return HNumber(value)


def float_to_hns(value: float, precision: int = 0) -> HNumber:
    """
    Convert float to HNumber with optional fractional scaling.
    
    Args:
        value: Float value to convert
        precision: Decimal places to preserve (0-6)
                   e.g., precision=2 means 1.23 → 123
    
    Returns:
        HNumber representing scaled value
    """
    scaled = value * (10 ** precision)
    return HNumber(int(round(scaled)))


def hns_to_integer(n: HNumber) -> int:
    """Convert HNumber back to integer."""
    return n.to_integer()


def hns_to_float(n: HNumber, precision: int = 0) -> float:
    """
    Convert HNumber back to float with precision unscaling.
    
    Args:
        n: HNumber to convert
        precision: Decimal places that were scaled (0-6)
    
    Returns:
        Float value
    """
    return n.to_integer() / (10 ** precision)


# ============================================================================
# Test Suite
# ============================================================================

def run_hns_tests():
    """Run comprehensive HNS validation tests."""
    print("=" * 60)
    print("HIERARCHICAL NUMBER SYSTEM - VESELOV/ANGULO")
    print("Validation Test Suite")
    print("=" * 60)
    
    # Test 1: Basic carry propagation
    print("\nTest 1: Carry Propagation (999,999 + 1 = 1,000,000)")
    a = HNumber([999.0, 999.0, 0.0, 0.0])  # 999,999
    b = HNumber([1.0, 0.0, 0.0, 0.0])       # 1
    result = hns_add(a, b)
    expected = [0.0, 0.0, 1.0, 0.0]  # 1,000,000
    
    print(f"  Input A: {a}")
    print(f"  Input B: {b}")
    print(f"  Result:  {result}")
    print(f"  Integer: {result.to_integer():,}")
    
    if result == expected:
        print("  ✓ SUCCESS: Cascading carry propagation works correctly")
    else:
        print("  ✗ FAILURE: Carry propagation error")
    
    # Test 2: Large number addition
    print("\nTest 2: Large Number Addition")
    a = HNumber([500.0, 500.0, 500.0, 0.0])  # 500,500,500
    b = HNumber([600.0, 600.0, 600.0, 0.0])  # 600,600,600
    result = hns_add(a, b)
    expected_value = 500500500 + 600600600  # 1,101,101,100
    
    print(f"  {a.to_integer():,} + {b.to_integer():,} = {result.to_integer():,}")
    
    if result.to_integer() == expected_value:
        print("  ✓ SUCCESS: Large addition correct")
    else:
        print(f"  ✗ FAILURE: Expected {expected_value:,}")
    
    # Test 3: Scaling (synaptic weight multiplication)
    print("\nTest 3: Synaptic Weight Scaling")
    activation = HNumber([0.0, 0.0, 1.0, 0.0])  # 1,000,000
    weight = 0.5
    result = hns_scale(activation, weight)
    
    print(f"  Activation: {activation.to_integer():,}")
    print(f"  Weight: {weight}")
    print(f"  Result: {result.to_integer():,}")
    
    if result.to_integer() == 500000:
        print("  ✓ SUCCESS: Synaptic scaling correct")
    else:
        print("  ✗ FAILURE: Scaling error")
    
    # Test 4: Precision preservation
    print("\nTest 4: Precision Over Many Operations")
    value = HNumber([1.0, 0.0, 0.0, 0.0])  # 1
    for i in range(1000):
        value = hns_add(value, HNumber([1.0, 0.0, 0.0, 0.0]))
    
    print(f"  1 + 1 (repeated 1000 times) = {value.to_integer():,}")
    
    if value.to_integer() == 1001:
        print("  ✓ SUCCESS: No precision loss after 1000 additions")
    else:
        print("  ✗ FAILURE: Precision loss detected")
    
    # Test 5: Batch operations
    print("\nTest 5: Batch Operations (1000 neurons)")
    neurons = np.random.randint(0, 1000, size=(1000, 4)).astype(np.float32)
    weights = np.random.uniform(0, 1, size=1000).astype(np.float32)
    
    # Normalize batch
    normalized = hns_normalize_batch(neurons)
    
    # Scale batch
    scaled = hns_scale_batch(normalized, weights)
    
    # Verify all values are in valid range
    valid = np.all(scaled[:, :3] < BASE) and np.all(scaled[:, :3] >= 0)
    
    print(f"  Processed 1000 neurons in batch")
    print(f"  All values in valid range: {valid}")
    
    if valid:
        print("  ✓ SUCCESS: Batch operations correct")
    else:
        print("  ✗ FAILURE: Invalid values in batch")
    
    # Test 6: Integer conversion roundtrip
    print("\nTest 6: Conversion Roundtrip")
    test_values = [0, 1, 999, 1000, 999999, 1000000, 999999999, 1234567890]
    all_passed = True
    
    for val in test_values:
        hn = HNumber(val)
        recovered = hn.to_integer()
        if recovered != val:
            print(f"  ✗ FAILURE: {val} → {recovered}")
            all_passed = False
    
    if all_passed:
        print(f"  ✓ SUCCESS: All {len(test_values)} conversions correct")
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_hns_tests()
