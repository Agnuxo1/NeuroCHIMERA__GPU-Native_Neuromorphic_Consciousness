# GPU Stress Test Report: RTX 3090 Maximum Capacity

**Generated:** 2025-12-01 16:38:39

## Executive Summary

This stress test determined the maximum network size that can be processed on an NVIDIA GeForce RTX 3090 GPU running NeuroCHIMERA.

**Maximum Successful Network Size:** 67,108,864 neurons

- **Texture Size:** 8192×8192
- **Memory Usage:** 3076.0 MB
- **Performance:** 30.25M neurons/s
- **Compute:** 0.76 GFLOPS

## Detailed Test Results

### Successful Tests

| Neurons | Texture | Memory (MB) | Time/Step (ms) | Throughput (M/s) | GFLOPS |
|---------|---------|-------------|----------------|------------------|--------|
| 2,097,152 | 2048×2048 | 196.0 | 191.25 | 10.97 | 0.27 |
| 4,194,304 | 2048×2048 | 196.0 | 156.24 | 26.85 | 0.67 |
| 8,388,608 | 4096×4096 | 772.0 | 596.00 | 14.07 | 0.35 |
| 16,777,216 | 4096×4096 | 772.0 | 589.93 | 28.44 | 0.71 |
| 33,554,432 | 8192×8192 | 3076.0 | 2079.69 | 16.13 | 0.40 |
| 67,108,864 | 8192×8192 | 3076.0 | 2218.71 | 30.25 | 0.76 |

## Performance Analysis

**Peak Throughput:** 30.25M neurons/s at 67,108,864 neurons

### Memory Efficiency

Memory usage scales approximately linearly with network size.

### Performance Scaling

Performance characteristics:

- **Large networks (≥1M):** Average 21.12M neurons/s
## Conclusions

1. **Maximum Viable Network:** 67,108,864 neurons (8192×8192 texture)
2. **Memory Limit:** Approximately 3076.0 MB for maximum network
3. **Performance at Maximum:** 30.25M neurons/s
4. **Peak Performance:** 30.25M neurons/s at 67,108,864 neurons

