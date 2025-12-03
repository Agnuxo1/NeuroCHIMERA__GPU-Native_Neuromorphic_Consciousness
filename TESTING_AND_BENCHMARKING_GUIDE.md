# NeuroCHIMERA Testing and Benchmarking Guide

This guide explains how to use the comprehensive testing and benchmarking suite for the NeuroCHIMERA system.

## Overview

The testing suite includes:
- **Core Component Tests**: Validation of HNS, Engine, Frame, Evolution, and Memory
- **Integration Tests**: Full system cycle validation
- **Consciousness Parameter Tests**: Validation of consciousness metrics
- **Performance Benchmarks**: HNS, system, and comparative benchmarks
- **Execution Scripts**: Long-term simulations and complete system benchmarks
- **Report Generators**: Automated professional report generation
- **Visualizations**: Plot generation for all metrics

## Quick Start

### Run All Tests

```bash
# Run core component tests
python tests/test_core_components.py

# Run integration tests
python tests/test_integration.py

# Run consciousness parameter tests
python tests/test_consciousness_parameters.py
```

### Run Benchmarks

```bash
# HNS performance benchmarks
python benchmarks/benchmark_hns_comprehensive.py

# System performance benchmarks
python benchmarks/benchmark_neurochimera_system.py

# Comparative benchmarks (requires PyTorch)
python benchmarks/benchmark_comparative.py
```

### Run Simulations

```bash
# Consciousness emergence simulation (1000 epochs, 65K neurons)
python run_consciousness_emergence.py --neurons 65536 --epochs 1000

# Complete system benchmark
python benchmark_complete_system.py --neurons 1048576 --epochs 100
```

### Generate Reports

```bash
# Generate all benchmark reports
python generate_benchmark_report.py

# Generate consciousness parameter reports
python generate_consciousness_report.py

# Generate visualizations
python create_visualizations.py

# Generate comprehensive execution results report
python document_execution_results.py
```

## Test Suites

### Core Component Tests (`tests/test_core_components.py`)

Validates:
- HNS operations (addition, scaling, normalization, multiplication)
- Engine initialization and GPU context
- Neuromorphic frame management
- Evolution dynamics
- Memory system operations

**Usage:**
```bash
python tests/test_core_components.py
```

### Integration Tests (`tests/test_integration.py`)

Validates:
- Complete evolution cycles
- Consciousness monitoring integration
- State persistence (save/load)
- Multi-epoch simulations
- Error recovery

**Usage:**
```bash
python tests/test_integration.py
```

### Consciousness Parameter Tests (`tests/test_consciousness_parameters.py`)

Validates:
- Critical parameter tracking (⟨k⟩, Φ, D, C, QCM)
- Phase transition detection
- Ethical monitoring
- Parameter evolution tracking

**Usage:**
```bash
python tests/test_consciousness_parameters.py
```

## Benchmark Scripts

### HNS Comprehensive Benchmarks (`benchmarks/benchmark_hns_comprehensive.py`)

Tests:
- Precision with large numbers
- Accumulative precision (1M+ iterations)
- Operation speed (addition, scaling)
- Batch operations

**Output:** `benchmarks/hns_benchmark_results.json`

### System Performance Benchmarks (`benchmarks/benchmark_neurochimera_system.py`)

Tests:
- Evolution speed for different network sizes
- Memory efficiency
- Scalability analysis
- Operations throughput

**Output:** `benchmarks/system_benchmark_results.json`

### Comparative Benchmarks (`benchmarks/benchmark_comparative.py`)

Compares CHIMERA vs PyTorch:
- Matrix operations
- Memory footprint
- Synaptic updates

**Requirements:** PyTorch (optional)

**Output:** `benchmarks/comparative_benchmark_results.json`

## Execution Scripts

### Consciousness Emergence Simulation (`run_consciousness_emergence.py`)

Runs long-term evolution to observe consciousness parameter evolution.

**Usage:**
```bash
# Default: 65K neurons, 1000 epochs
python run_consciousness_emergence.py

# Custom configuration
python run_consciousness_emergence.py --neurons 1048576 --epochs 5000 --iterations 20

# Disable HNS
python run_consciousness_emergence.py --no-hns
```

**Output:** `results/consciousness_emergence_YYYYMMDD_HHMMSS.json`

### Complete System Benchmark (`benchmark_complete_system.py`)

Comprehensive system performance benchmark.

**Usage:**
```bash
python benchmark_complete_system.py --neurons 1048576 --epochs 100
```

**Output:** `results/complete_system_benchmark_YYYYMMDD_HHMMSS.json`

## Report Generation

### Benchmark Reports (`generate_benchmark_report.py`)

Generates professional English reports from benchmark results:
- HNS Performance Benchmark Report
- System Performance Report
- Comparative Benchmark Report

**Output:** `reports/HNS_BENCHMARK_REPORT.md`, `reports/SYSTEM_PERFORMANCE_REPORT.md`, `reports/COMPARATIVE_BENCHMARK_REPORT.md`

### Consciousness Reports (`generate_consciousness_report.py`)

Generates consciousness parameter analysis reports:
- Parameter evolution analysis
- Threshold crossing detection
- Critical event documentation

**Output:** `reports/consciousness_analysis_*.md`

### Visualizations (`create_visualizations.py`)

Generates plots and graphs:
- Consciousness parameter evolution
- Performance benchmarks
- Comparative benchmarks

**Requirements:** matplotlib

**Output:** `visualizations/*.png`

### Comprehensive Documentation (`document_execution_results.py`)

Generates master execution results report integrating all findings.

**Output:** `reports/EXECUTION_RESULTS.md`

## Directory Structure

```
Vladimir/
├── tests/                          # Test suites
│   ├── test_core_components.py
│   ├── test_integration.py
│   └── test_consciousness_parameters.py
├── benchmarks/                      # Benchmark scripts
│   ├── benchmark_hns_comprehensive.py
│   ├── benchmark_neurochimera_system.py
│   ├── benchmark_comparative.py
│   └── *.json                      # Benchmark results
├── results/                        # Simulation results
│   └── consciousness_emergence_*.json
├── reports/                         # Generated reports
│   ├── HNS_BENCHMARK_REPORT.md
│   ├── SYSTEM_PERFORMANCE_REPORT.md
│   ├── COMPARATIVE_BENCHMARK_REPORT.md
│   ├── consciousness_analysis_*.md
│   └── EXECUTION_RESULTS.md
└── visualizations/                  # Generated plots
    └── *.png
```

## Complete Workflow

To run the complete testing and benchmarking workflow:

1. **Run Tests**
   ```bash
   python tests/test_core_components.py
   python tests/test_integration.py
   python tests/test_consciousness_parameters.py
   ```

2. **Run Benchmarks**
   ```bash
   python benchmarks/benchmark_hns_comprehensive.py
   python benchmarks/benchmark_neurochimera_system.py
   python benchmarks/benchmark_comparative.py  # Optional, requires PyTorch
   ```

3. **Run Simulations**
   ```bash
   python run_consciousness_emergence.py --neurons 65536 --epochs 1000
   python benchmark_complete_system.py --neurons 1048576 --epochs 100
   ```

4. **Generate Reports**
   ```bash
   python generate_benchmark_report.py
   python generate_consciousness_report.py
   python create_visualizations.py
   python document_execution_results.py
   ```

5. **Review Results**
   - Check `reports/EXECUTION_RESULTS.md` for comprehensive summary
   - Review individual reports in `reports/` directory
   - View visualizations in `visualizations/` directory

## Requirements

### Core Dependencies
- Python 3.8+
- numpy
- moderngl (for GPU support)

### Optional Dependencies
- PyTorch (for comparative benchmarks)
- matplotlib (for visualizations)
- psutil (for memory monitoring)

## Notes

- GPU benchmarks require OpenGL 4.3+ compatible GPU
- Some benchmarks may take significant time for large networks
- Comparative benchmarks are optional and require PyTorch
- Visualizations require matplotlib
- All reports are generated in English as specified

## Troubleshooting

### GPU Not Available
If GPU initialization fails, the system will automatically fall back to CPU mode. Tests and benchmarks will still run, but performance will be slower.

### Missing Dependencies
Install missing dependencies:
```bash
pip install numpy moderngl matplotlib psutil
# Optional:
pip install torch  # For comparative benchmarks
```

### Memory Issues
For large networks, reduce network size or use smaller epoch counts:
```bash
python run_consciousness_emergence.py --neurons 65536 --epochs 100
```

## Support

For issues or questions, refer to the main README.md or check the generated reports for detailed analysis.

