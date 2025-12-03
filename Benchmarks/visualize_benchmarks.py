"""
Benchmark Visualization Suite
==============================
Creates publication-quality graphs from benchmark JSON results.

Generates:
- Performance comparison charts
- Speedup visualizations
- Statistical confidence intervals
- Memory efficiency graphs
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300


class BenchmarkVisualizer:
    """Creates publication-quality visualizations from benchmark results."""

    def __init__(self, output_dir: str = "benchmark_graphs"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save generated graphs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"Saving graphs to: {self.output_dir}/")

    def visualize_gpu_hns_benchmarks(self, json_file: str):
        """Visualize GPU HNS benchmark results."""
        print(f"\nVisualizing GPU HNS benchmarks from {json_file}...")

        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract data
        addition_results = data["results"]["addition"]
        scaling_results = data["results"]["scaling"]

        sizes = [r["size"] for r in addition_results]
        add_throughput = [r["throughput_ops_per_sec"] for r in addition_results]
        add_std = [r["std_time_ms"] for r in addition_results]
        scale_throughput = [r["throughput_ops_per_sec"] for r in scaling_results]
        scale_std = [r["std_time_ms"] for r in scaling_results]

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Throughput comparison
        x = np.arange(len(sizes))
        width = 0.35

        ax1.bar(x - width/2, np.array(add_throughput)/1e6, width,
                label='Addition', alpha=0.8, color='#2E86AB')
        ax1.bar(x + width/2, np.array(scale_throughput)/1e6, width,
                label='Scaling', alpha=0.8, color='#A23B72')

        ax1.set_xlabel('Problem Size (Operations)')
        ax1.set_ylabel('Throughput (Million ops/sec)')
        ax1.set_title('GPU HNS Performance: Addition vs Scaling')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{s:,}' for s in sizes], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Execution time with error bars
        add_times = [r["mean_time_ms"] for r in addition_results]
        scale_times = [r["mean_time_ms"] for r in scaling_results]

        ax2.errorbar(sizes, add_times, yerr=add_std, marker='o',
                     label='Addition', capsize=5, capthick=2, linewidth=2,
                     color='#2E86AB')
        ax2.errorbar(sizes, scale_times, yerr=scale_std, marker='s',
                     label='Scaling', capsize=5, capthick=2, linewidth=2,
                     color='#A23B72')

        ax2.set_xlabel('Problem Size (Operations)')
        ax2.set_ylabel('Execution Time (ms)')
        ax2.set_title('GPU HNS Execution Time (with std dev)')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        output_file = self.output_dir / "gpu_hns_performance.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: {output_file}")

    def visualize_comparative_benchmarks(self, json_file: str):
        """Visualize framework comparison results."""
        print(f"\nVisualizing comparative benchmarks from {json_file}...")

        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract comparison data
        sizes = data["configuration"]["matrix_sizes"]

        # Prepare data for plotting
        frameworks_data = {}

        for size in sizes:
            size_key = f"size_{size}"
            if size_key not in data["comparison"]:
                continue

            for comp in data["comparison"][size_key]:
                fw_name = f"{comp['framework']} ({comp['device']})"
                if fw_name not in frameworks_data:
                    frameworks_data[fw_name] = {
                        'sizes': [],
                        'gflops': [],
                        'speedup': []
                    }

                frameworks_data[fw_name]['sizes'].append(size)
                frameworks_data[fw_name]['gflops'].append(comp['gflops'])
                frameworks_data[fw_name]['speedup'].append(comp['speedup_vs_numpy'])

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: GFLOPS comparison
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        for (fw_name, fw_data), color in zip(frameworks_data.items(), colors):
            ax1.plot(fw_data['sizes'], fw_data['gflops'], marker='o',
                    linewidth=2, markersize=8, label=fw_name, color=color)

        ax1.set_xlabel('Matrix Size (NxN)')
        ax1.set_ylabel('Performance (GFLOPS)')
        ax1.set_title('Matrix Multiplication Performance Comparison')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')

        # Plot 2: Speedup vs NumPy
        for (fw_name, fw_data), color in zip(frameworks_data.items(), colors):
            if 'NumPy' not in fw_name:  # Skip NumPy itself
                ax2.bar(fw_data['sizes'], fw_data['speedup'],
                       alpha=0.7, label=fw_name, color=color)

        ax2.set_xlabel('Matrix Size (NxN)')
        ax2.set_ylabel('Speedup vs NumPy (x)')
        ax2.set_title('Framework Speedup Comparison (Baseline: NumPy CPU)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticklabels([f'{s}' for s in sizes])

        plt.tight_layout()
        output_file = self.output_dir / "framework_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: {output_file}")

    def visualize_hns_cpu_benchmarks(self, json_file: str):
        """Visualize CPU HNS benchmark results."""
        print(f"\nVisualizing HNS CPU benchmarks from {json_file}...")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"  [FAILED] File not found: {json_file}")
            return

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Speed comparison (if available)
        if "speed" in data:
            operations = ['Addition', 'Scaling']
            float_times = [
                data["speed"]["add"]["float"] * 1000,
                data["speed"]["scale"]["float"] * 1000
            ]
            hns_times = [
                data["speed"]["add"]["hns"] * 1000,
                data["speed"]["scale"]["hns"] * 1000
            ]

            x = np.arange(len(operations))
            width = 0.35

            ax1.bar(x - width/2, float_times, width, label='Float',
                   alpha=0.8, color='#2E86AB')
            ax1.bar(x + width/2, hns_times, width, label='HNS',
                   alpha=0.8, color='#A23B72')

            ax1.set_ylabel('Execution Time (ms)')
            ax1.set_title('HNS vs Float: Operation Speed')
            ax1.set_xticks(x)
            ax1.set_xticklabels(operations)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Accumulative precision
        if "accumulative" in data:
            frameworks = ['Float', 'HNS', 'Decimal']
            errors = [
                data["accumulative"]["float"]["error"],
                data["accumulative"]["hns"]["error"],
                data["accumulative"].get("decimal", {}).get("error", 0)
            ]

            colors = ['#2E86AB', '#A23B72', '#F18F01']
            ax2.bar(frameworks, errors, alpha=0.8, color=colors)
            ax2.set_ylabel('Absolute Error')
            ax2.set_title(f'Accumulative Precision\n({data["accumulative"]["iterations"]:,} iterations)')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3, axis='y')

            # Add "PASSED" annotation for HNS if error < 1e-9
            if data["accumulative"]["hns"]["error"] < 1e-9:
                ax2.text(1, errors[1], 'PASSED [OK]',
                        ha='center', va='bottom', fontsize=12, color='green',
                        fontweight='bold')

        # Plot 3: Overhead comparison
        if "speed" in data:
            operations = ['Addition', 'Scaling']
            overheads = [
                data["speed"]["add"]["overhead"],
                data["speed"]["scale"]["overhead"]
            ]

            ax3.bar(operations, overheads, alpha=0.8, color='#C73E1D')
            ax3.set_ylabel('Overhead (x times slower)')
            ax3.set_title('HNS CPU Overhead vs Float')
            ax3.grid(True, alpha=0.3, axis='y')

            # Add reference line at 200x
            ax3.axhline(y=200, color='red', linestyle='--',
                       label='~200x (documented)', linewidth=2)
            ax3.legend()

        plt.tight_layout()
        output_file = self.output_dir / "hns_cpu_benchmarks.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: {output_file}")

    def create_summary_dashboard(self, json_files: Dict[str, str]):
        """
        Create a comprehensive dashboard with all benchmarks.

        Args:
            json_files: Dictionary mapping benchmark names to JSON file paths
        """
        print("\nCreating summary dashboard...")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('NeuroCHIMERA Benchmark Summary Dashboard',
                    fontsize=20, fontweight='bold')

        # Try to load and plot each benchmark type
        # This will be expanded based on available data

        output_file = self.output_dir / "benchmark_dashboard.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: {output_file}")


def main():
    """Generate all benchmark visualizations."""
    print("=" * 80)
    print("BENCHMARK VISUALIZATION SUITE")
    print("=" * 80)

    visualizer = BenchmarkVisualizer(output_dir="benchmark_graphs")

    # Find all JSON benchmark files
    benchmark_files = {
        "gpu_hns": "gpu_hns_complete_benchmark_results.json",
        "comparative": "comparative_benchmark_results.json",
        "hns_cpu": "hns_benchmark_results.json",
    }

    generated_count = 0

    # Visualize GPU HNS benchmarks
    if Path(benchmark_files["gpu_hns"]).exists():
        try:
            visualizer.visualize_gpu_hns_benchmarks(benchmark_files["gpu_hns"])
            generated_count += 1
        except Exception as e:
            print(f"  [FAILED] Error visualizing GPU HNS: {e}")

    # Visualize comparative benchmarks
    if Path(benchmark_files["comparative"]).exists():
        try:
            visualizer.visualize_comparative_benchmarks(benchmark_files["comparative"])
            generated_count += 1
        except Exception as e:
            print(f"  [FAILED] Error visualizing comparative: {e}")

    # Visualize HNS CPU benchmarks
    if Path(benchmark_files["hns_cpu"]).exists():
        try:
            visualizer.visualize_hns_cpu_benchmarks(benchmark_files["hns_cpu"])
            generated_count += 1
        except Exception as e:
            print(f"  [FAILED] Error visualizing HNS CPU: {e}")

    print("\n" + "=" * 80)
    print(f"VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\n[OK] Generated {generated_count} visualization(s)")
    print(f"[OK] Saved to: benchmark_graphs/")
    print("\nFiles created:")

    for png_file in Path("benchmark_graphs").glob("*.png"):
        print(f"  - {png_file.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
