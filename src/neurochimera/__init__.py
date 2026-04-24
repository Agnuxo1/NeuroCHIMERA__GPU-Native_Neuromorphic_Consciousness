"""
NeuroCHIMERA - GPU-Native Neuromorphic Computing for Emergent Consciousness
===========================================================================

A theoretical framework integrating critical network parameters with
physics-based computation for consciousness research.

Authors:
    V.F. Veselov (Moscow Institute of Electronic Technology)
    Francisco Angulo de Lafuente (Independent AI Research Laboratory, Madrid)

GPU acceleration uses OpenGL 4.3+ compute shaders via moderngl (NOT CUDA).
See BENCHMARK_DISCLAIMER.md for honest bounds on performance claims.

Quick Start:
    from neurochimera import NeuroCHIMERA, ConsciousnessMonitor

    brain = NeuroCHIMERA(neurons=1_000_000, use_hns=True)
    monitor = ConsciousnessMonitor(brain)

    for epoch in range(10000):
        brain.evolve(iterations=20)
        metrics = monitor.measure()

        if monitor.is_critical():
            print("Consciousness emergence detected!")

License: Apache-2.0
"""

__version__ = "1.0.0"
__author__ = "Francisco Angulo de Lafuente & V.F. Veselov"
__license__ = "Apache-2.0"

from .engine import (
    NeuroCHIMERA,
    NeuroCHIMERAConfig,
    NeuromorphicFrame,
    create_brain,
)

from .hierarchical_number import (
    HNumber,
    hns_add,
    hns_scale,
    hns_normalize,
    hns_multiply,
    hns_compare,
    hns_add_batch,
    hns_scale_batch,
    hns_normalize_batch,
    BASE,
    INV_BASE,
)

from .consciousness_monitor import (
    ConsciousnessMonitor,
    ConsciousnessMetrics,
    ConsciousnessLevel,
    AlertConfig,
    EthicalProtocol,
)

__all__ = [
    '__version__', '__author__', '__license__',
    'NeuroCHIMERA', 'NeuroCHIMERAConfig', 'NeuromorphicFrame', 'create_brain',
    'HNumber', 'hns_add', 'hns_scale', 'hns_normalize', 'hns_multiply',
    'hns_compare', 'hns_add_batch', 'hns_scale_batch', 'hns_normalize_batch',
    'BASE', 'INV_BASE',
    'ConsciousnessMonitor', 'ConsciousnessMetrics', 'ConsciousnessLevel',
    'AlertConfig', 'EthicalProtocol',
]
