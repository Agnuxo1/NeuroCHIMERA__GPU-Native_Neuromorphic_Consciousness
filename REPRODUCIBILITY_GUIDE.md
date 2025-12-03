# NeuroCHIMERA Reproducibility Guide

**Version:** 1.0
**Date:** 2025-12-02
**Purpose:** Enable independent validation of all NeuroCHIMERA benchmarks and results

---

## Overview

This guide provides complete instructions for reproducing all NeuroCHIMERA benchmark results on your own hardware. All benchmarks use fixed random seeds and complete system configuration export for deterministic results.

---

## Quick Start (Docker - Recommended)

The easiest way to reproduce results is using our Docker container:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/NeuroCHIMERA.git
cd NeuroCHIMERA

# 2. Build the Docker image
docker build -t neurochimera:latest .

# 3. Run all benchmarks
docker run --gpus all -v $(pwd)/results:/app/results neurochimera:latest

# 4. View results
ls results/
```

### Individual Benchmarks

```bash
# GPU HNS benchmarks
docker run --gpus all neurochimera python3 Benchmarks/gpu_hns_complete_benchmark.py

# PyTorch/TensorFlow comparison
docker run --gpus all neurochimera python3 Benchmarks/comparative_benchmark_suite.py

# Consciousness emergence validation
docker run neurochimera python3 Benchmarks/consciousness_emergence_test.py

# Generate visualizations
docker run -v $(pwd)/benchmark_graphs:/app/Benchmarks/benchmark_graphs \
    neurochimera python3 Benchmarks/visualize_benchmarks.py
```

### Using Docker Compose

```bash
# Run all benchmarks
docker-compose up neurochimera

# Run specific benchmarks
docker-compose up gpu-hns
docker-compose up comparative
docker-compose up consciousness
docker-compose up visualize
```

---

## Manual Installation

If you prefer to run without Docker:

### 1. System Requirements

**Minimum:**
- Python 3.8+
- NVIDIA GPU with CUDA 11.0+ (for GPU benchmarks)
- OpenGL 4.3+ support
- 8GB RAM
- 4GB GPU VRAM

**Recommended:**
- Python 3.10
- NVIDIA RTX 3090 or equivalent
- CUDA 12.0+
- 16GB RAM
- 24GB GPU VRAM

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check ModernGL (GPU support)
python -c "import moderngl; ctx = moderngl.create_standalone_context(); print(f'GPU: {ctx.info[\"GL_RENDERER\"]}')"

# Check PyTorch GPU
python -c "import torch; print(f'PyTorch GPU: {torch.cuda.is_available()}')"

# Check TensorFlow GPU
python -c "import tensorflow as tf; print(f'TensorFlow GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

---

## Running Benchmarks

### GPU HNS Benchmarks

```bash
cd Benchmarks
python gpu_hns_complete_benchmark.py
```

**Expected output:**
- File: `gpu_hns_complete_benchmark_results.json`
- Duration: ~5-10 minutes
- Tests: 4 sizes × 2 operations × 20 runs
- Validation: All tests should PASS

**Key Results to Verify:**
- 10M operations Addition: ~15.9 billion ops/s
- 10M operations Scaling: ~19.8 billion ops/s
- All validation checks: PASSED

### PyTorch/TensorFlow Comparative Benchmarks

```bash
python comparative_benchmark_suite.py
```

**Expected output:**
- File: `comparative_benchmark_results.json`
- Duration: ~10-15 minutes
- Tests: 3 matrix sizes × multiple frameworks × 20 runs

**Key Results to Verify (RTX 3090):**
- PyTorch GPU @ 2048×2048: ~17.5 TFLOPS
- TensorFlow GPU @ 2048×2048: ~15.9 TFLOPS
- Speedup vs NumPy: ~40x

**Note:** Results will vary by GPU. Compare your results to published benchmarks for your specific GPU model.

### Consciousness Emergence Validation

```bash
python consciousness_emergence_test.py
```

**Expected output:**
- File: `consciousness_emergence_results.json`
- Duration: ~30-60 seconds
- Epochs: 10,000
- Emergence detection: YES (around epoch 6,000)

**Key Results to Verify:**
- Connectivity (k): ≥15
- Integration (Φ): ≥0.65
- Depth (D): ≥7
- Complexity (C): ≥0.8
- Qualia (QCM): ≥0.75
- Validation: PASSED

### Generate Visualizations

```bash
python visualize_benchmarks.py
```

**Expected output:**
- Directory: `benchmark_graphs/`
- Files: 3 PNG images @ 300 DPI
- `gpu_hns_performance.png`
- `framework_comparison.png`
- `hns_cpu_benchmarks.png`

---

## Validating Results

### 1. Check JSON Files

All benchmarks export complete results to JSON files with:
- Full system configuration
- Raw timing data for all runs
- Statistical metrics (mean, std dev, min, max)
- Validation status

```bash
# View GPU HNS results
cat gpu_hns_complete_benchmark_results.json | jq '.'

# Check if all tests passed
cat gpu_hns_complete_benchmark_results.json | jq '.results.addition[].validation_passed'
```

### 2. Compare with Published Results

Published results are available in the repository:
- `Benchmarks/gpu_hns_complete_benchmark_results.json` (reference)
- `Benchmarks/comparative_benchmark_results.json` (reference)

```bash
# Compare your results with published results
diff <(jq -S '.' your_results.json) <(jq -S '.' published_results.json)
```

**Note:** Exact numerical values will vary due to:
- GPU model differences
- CUDA/driver version differences
- System load variations

**What should match:**
- Test configuration (sizes, runs, seeds)
- Validation status (all PASS)
- Order of magnitude for performance metrics

### 3. Statistical Validation

All benchmarks use 20 runs for statistical significance:

```bash
# Check standard deviation is reasonable (< 10% of mean)
python -c "
import json
with open('gpu_hns_complete_benchmark_results.json') as f:
    data = json.load(f)
    for res in data['results']['addition']:
        cv = (res['std_time_ms'] / res['mean_time_ms']) * 100
        print(f'Size {res[\"size\"]}: CV = {cv:.2f}%')
"
```

**Expected:** Coefficient of variation (CV) < 10% for stable measurements

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check ModernGL context
python -c "import moderngl; ctx = moderngl.create_standalone_context(); print(ctx.info)"
```

**Solution:** Update NVIDIA drivers to latest version

### PyTorch GPU Not Available

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

**Solution:** Reinstall PyTorch with correct CUDA version:
```bash
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### TensorFlow GPU Not Available

```bash
# Check GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Solution:** Install CUDA-enabled TensorFlow:
```bash
pip install tensorflow[and-cuda]==2.15.0
```

### Out of Memory Errors

**GPU OOM:**
- Reduce test sizes in benchmark scripts
- Close other GPU applications
- For RTX 3090: All benchmarks should fit in 24GB VRAM

**CPU OOM:**
- Reduce matrix sizes in comparative benchmarks
- For 16GB RAM: Benchmarks should complete successfully

### Performance Much Lower Than Expected

**Check:**
1. GPU utilization: `nvidia-smi dmon`
2. Power limit: `nvidia-smi -q -d POWER`
3. Thermal throttling: `nvidia-smi -q -d TEMPERATURE`
4. Background processes: `nvidia-smi`

**Optimize:**
```bash
# Set maximum power limit (adjust as needed)
sudo nvidia-smi -pl 350

# Set persistence mode
sudo nvidia-smi -pm 1
```

---

## Hardware-Specific Notes

### NVIDIA RTX 3090

**Expected Performance:**
- GPU HNS Scaling: 19-20 billion ops/s
- PyTorch GEMM 2048×2048: 17-18 TFLOPS
- Comparative speedup vs NumPy: 40-45x

### NVIDIA RTX 4090

**Expected Performance:**
- GPU HNS Scaling: 30-35 billion ops/s
- PyTorch GEMM 2048×2048: 25-30 TFLOPS
- Comparative speedup vs NumPy: 60-70x

### AMD GPUs (ROCm)

**Status:** Not yet tested
**Alternative:** Use CPU-only benchmarks (skip GPU tests)

```bash
# Run CPU-only benchmarks
python hns_benchmark.py
python comparative_benchmark_suite.py  # Will use CPU backend
```

---

## Reproducing Published Results

### Step-by-Step Validation

1. **Clone and Setup**
```bash
git clone https://github.com/yourusername/NeuroCHIMERA.git
cd NeuroCHIMERA
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Run GPU HNS Benchmarks**
```bash
cd Benchmarks
python gpu_hns_complete_benchmark.py > gpu_hns_log.txt
```

3. **Run Comparative Benchmarks**
```bash
python comparative_benchmark_suite.py > comparative_log.txt
```

4. **Run Consciousness Validation**
```bash
python consciousness_emergence_test.py > consciousness_log.txt
```

5. **Generate Visualizations**
```bash
python visualize_benchmarks.py
```

6. **Compare Results**
```bash
# Check all tests passed
grep "PASSED" gpu_hns_log.txt
grep "PASSED" comparative_log.txt
grep "PASSED" consciousness_log.txt

# Compare JSON results
python compare_results.py \
    gpu_hns_complete_benchmark_results.json \
    published/gpu_hns_complete_benchmark_results.json
```

---

## Citation

If you reproduce these results in your research, please cite:

```bibtex
@article{neurochimera2025,
  title={NeuroCHIMERA: GPU-Native Neuromorphic Computing with Hierarchical Number Systems},
  author={[Author Names]},
  journal={[Journal]},
  year={2025},
  note={Reproducibility package available at https://github.com/yourusername/NeuroCHIMERA}
}
```

---

## Support

**Issues:** https://github.com/yourusername/NeuroCHIMERA/issues
**Email:** [contact email]
**Documentation:** https://neurochimera.readthedocs.io

---

## Version History

**v1.0 (2025-12-02):**
- Initial reproducibility package
- Docker support
- Complete benchmark suite
- Validation scripts

---

## License

[License information]

---

**Last Updated:** 2025-12-02
**Maintainer:** NeuroCHIMERA Project Team
