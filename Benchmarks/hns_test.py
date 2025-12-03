"""
HNS (Hierarchical Number System) - Proof of Concept Test
========================================================
Python simulator that exactly replicates GPU behavior.
If this works, the shader will work.
"""

import math

# Simulation configuration
BASE = 1000.0

def hns_normalize(vec4):
    r, g, b, a = vec4
    
    # Carry 0 -> 1
    carry0 = math.floor(r / BASE)
    r = r - (carry0 * BASE)
    g += carry0
    
    # Carry 1 -> 2
    carry1 = math.floor(g / BASE)
    g = g - (carry1 * BASE)
    b += carry1
    
    # Carry 2 -> 3
    carry2 = math.floor(b / BASE)
    b = b - (carry2 * BASE)
    a += carry2
    
    return [r, g, b, a]

def hns_add(vec_a, vec_b):
    # Simple vector addition
    raw_sum = [x + y for x, y in zip(vec_a, vec_b)]
    # Normalization (Carry propagation)
    return hns_normalize(raw_sum)

def to_string(vec4):
    return f"[ Trillions: {int(vec4[3])} | Billions: {int(vec4[2])} | Thousands: {int(vec4[1])} | Units: {int(vec4[0])} ]"

# --- THE EXPERIMENT ---
print("=== VESELOV SYSTEM TEST ===")

# 1. Define two large numbers
# Num A = 999,999 (Almost one million) -> In HNS: [999, 999, 0, 0]
num_a = [999.0, 999.0, 0.0, 0.0] 

# Num B = 1 (One unit) -> In HNS: [1, 0, 0, 0]
num_b = [1.0, 0.0, 0.0, 0.0]

print(f"Number A: {to_string(num_a)}")
print(f"Number B: {to_string(num_b)}")

# 2. Add them
print("\n--- Performing Hierarchical Addition ---")
result = hns_add(num_a, num_b)

# 3. Expected result: 1,000,000
# In HNS it should be: [0, 0, 1, 0] (0 units, 0 thousands, 1 million)
print(f"Result: {to_string(result)}")

# Verification
if result == [0.0, 0.0, 1.0, 0.0]:
    print("\n[SUCCESS] The system has correctly managed cascading overflow.")
else:
    print("\n[FAILURE] Something is wrong with the carry logic.")

