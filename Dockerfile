# NeuroCHIMERA Reproducibility Docker Image
# ==========================================
# This Docker image provides a complete, reproducible environment
# for running all NeuroCHIMERA benchmarks and validations.
#
# Build: docker build -t neurochimera:latest .
# Run:   docker run --gpus all -v $(pwd)/results:/app/results neurochimera:latest

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL maintainer="NeuroCHIMERA Project"
LABEL description="Reproducible environment for NeuroCHIMERA benchmarks and validation"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    moderngl==5.8.2 \
    pillow==10.0.0 \
    matplotlib==3.10.0 \
    seaborn==0.13.2 \
    pytest==7.4.0 \
    torch==2.1.0 \
    tensorflow==2.15.0

# Copy project files
COPY . /app/

# Create results directory
RUN mkdir -p /app/results \
    && mkdir -p /app/Benchmarks/benchmark_graphs

# Set permissions
RUN chmod +x /app/Benchmarks/*.py

# Default command: run all benchmarks
CMD ["python3", "Benchmarks/run_all_benchmarks.py"]

# Alternative commands:
# - GPU HNS benchmarks: docker run --gpus all neurochimera python3 Benchmarks/gpu_hns_complete_benchmark.py
# - PyTorch comparison: docker run --gpus all neurochimera python3 Benchmarks/comparative_benchmark_suite.py
# - Consciousness test: docker run --gpus all neurochimera python3 Benchmarks/consciousness_emergence_test.py
# - Visualizations: docker run neurochimera python3 Benchmarks/visualize_benchmarks.py
