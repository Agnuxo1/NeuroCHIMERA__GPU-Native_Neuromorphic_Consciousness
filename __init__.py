"""
NeuroCHIMERA - GPU-Native Neuromorphic Computing for Emergent Consciousness
===========================================================================

A theoretical framework integrating critical network parameters with
physics-based computation for consciousness research.

Authors:
    V.F. Veselov (Moscow Institute of Electronic Technology)
    Francisco Angulo de Lafuente (Independent AI Research Laboratory, Madrid)

Main Components:
    NeuroCHIMERA: Main GPU-native neuromorphic engine
    ConsciousnessMonitor: Critical parameter tracking
    HNumber: Hierarchical Number System for extended precision
    
Quick Start:
    from neurochimera import NeuroCHIMERA, ConsciousnessMonitor
    
    brain = NeuroCHIMERA(neurons=1_000_000, use_hns=True)
    monitor = ConsciousnessMonitor(brain)
    
    for epoch in range(10000):
        brain.evolve(iterations=20)
        metrics = monitor.measure()
        
        if monitor.is_critical():
            print("Consciousness emergence detected!")

License: MIT
"""

__version__ = "1.0.0"
__author__ = "V.F. Veselov & Francisco Angulo de Lafuente"
__license__ = "MIT"

# Core imports
# NOTE:
# The original layout expected a package structure like:
#   core/engine.py, hns/hierarchical_number.py, metrics/consciousness_monitor.py
# In this repository the modules live at the top level (engine.py, hierarchical_number.py,
# consciousness_monitor.py). To keep compatibility we try the package layout first and
# fall back to the flat layout if needed.
try:
    from .core.engine import (
        NeuroCHIMERA,
        NeuroCHIMERAConfig,
        NeuromorphicFrame,
        create_brain,
    )
except ImportError:  # Fallback to flat layout
    from .engine import (
        NeuroCHIMERA,
        NeuroCHIMERAConfig,
        NeuromorphicFrame,
        create_brain,
    )

# HNS imports
try:
    from .hns.hierarchical_number import (
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
except ImportError:  # Fallback to flat layout
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

# Metrics imports
try:
    from .metrics.consciousness_monitor import (
        ConsciousnessMonitor,
        ConsciousnessMetrics,
        ConsciousnessLevel,
        AlertConfig,
        EthicalProtocol,
    )
except ImportError:  # Fallback to flat layout
    from .consciousness_monitor import (
        ConsciousnessMonitor,
        ConsciousnessMetrics,
        ConsciousnessLevel,
        AlertConfig,
        EthicalProtocol,
    )

# All public exports
__all__ = [
    # Version
    '__version__',
    '__author__',
    '__license__',
    
    # Core
    'NeuroCHIMERA',
    'NeuroCHIMERAConfig',
    'NeuromorphicFrame',
    'create_brain',
    
    # HNS
    'HNumber',
    'hns_add',
    'hns_scale',
    'hns_normalize',
    'hns_multiply',
    'hns_compare',
    'hns_add_batch',
    'hns_scale_batch',
    'hns_normalize_batch',
    'BASE',
    'INV_BASE',
    
    # Metrics
    'ConsciousnessMonitor',
    'ConsciousnessMetrics',
    'ConsciousnessLevel',
    'AlertConfig',
    'EthicalProtocol',
]
