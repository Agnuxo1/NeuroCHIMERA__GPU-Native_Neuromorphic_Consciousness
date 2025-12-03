"""
Real-Time GPU Utilization Monitor
==================================
Monitors GPU utilization during benchmarks to verify we're saturating the GPU.

Tracks:
- GPU utilization %
- Memory usage
- Temperature
- Power draw
- Clock speeds
"""

import subprocess
import time
import json
from datetime import datetime

def get_nvidia_smi_data():
    """Query NVIDIA GPU metrics using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=2)

        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            return {
                'gpu_util_percent': float(values[0]),
                'mem_util_percent': float(values[1]),
                'mem_used_mb': float(values[2]),
                'mem_total_mb': float(values[3]),
                'temp_celsius': float(values[4]),
                'power_draw_watts': float(values[5]),
                'gpu_clock_mhz': float(values[6]),
                'mem_clock_mhz': float(values[7]),
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}")

    return None


def monitor_continuous(duration_seconds=60, interval=0.5):
    """Monitor GPU for specified duration"""
    print(f"Monitoring GPU for {duration_seconds} seconds (interval: {interval}s)...")
    print("GPU%  Mem%  Temp  Power  GPU_MHz  Mem_MHz")
    print("-" * 50)

    samples = []
    start_time = time.time()
    end_time = start_time + duration_seconds

    while time.time() < end_time:
        data = get_nvidia_smi_data()
        if data:
            samples.append(data)
            print(f"{data['gpu_util_percent']:3.0f}%  {data['mem_util_percent']:3.0f}%  "
                  f"{data['temp_celsius']:3.0f}C  {data['power_draw_watts']:5.1f}W  "
                  f"{data['gpu_clock_mhz']:4.0f}MHz  {data['mem_clock_mhz']:4.0f}MHz")

        time.sleep(interval)

    # Calculate statistics
    if samples:
        gpu_utils = [s['gpu_util_percent'] for s in samples]
        avg_util = sum(gpu_utils) / len(gpu_utils)
        max_util = max(gpu_utils)
        min_util = min(gpu_utils)

        print("\n" + "=" * 50)
        print("GPU Utilization Statistics")
        print("=" * 50)
        print(f"Average: {avg_util:.1f}%")
        print(f"Maximum: {max_util:.1f}%")
        print(f"Minimum: {min_util:.1f}%")
        print(f"Samples: {len(samples)}")
        print("=" * 50)

        # Check if GPU is saturated
        if avg_util > 85:
            print("[OK] GPU is SATURATED (>85% average utilization)")
        elif avg_util > 50:
            print("[WARNING] GPU utilization moderate (50-85%)")
        else:
            print("[ISSUE] GPU underutilized (<50% average)")

        return {
            'samples': samples,
            'statistics': {
                'avg_utilization': avg_util,
                'max_utilization': max_util,
                'min_utilization': min_util,
                'sample_count': len(samples)
            }
        }

    return None


def main():
    print("GPU Utilization Monitor")
    print("=" * 50)
    print("This will monitor GPU utilization for 60 seconds.")
    print("Run your benchmark in another terminal simultaneously.")
    print()
    input("Press ENTER to start monitoring...")

    results = monitor_continuous(duration_seconds=60, interval=0.5)

    if results:
        # Save to JSON
        output_file = 'gpu_utilization_profile.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {output_file}")


if __name__ == '__main__':
    main()
