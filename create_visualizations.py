"""
Visualization Generator
======================

Generate plots and visualizations for:
- Metrics evolution
- Performance benchmarks
- Parameter evolution
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


class VisualizationGenerator:
    """Generate visualizations for benchmark and simulation results."""
    
    def __init__(self, results_dir: Path = None, output_dir: Path = None):
        if results_dir is None:
            results_dir = Path(__file__).parent / "results"
        if output_dir is None:
            output_dir = Path(__file__).parent / "visualizations"
        
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_consciousness_evolution(self, results: Dict, filename: str = None):
        """Plot consciousness parameter evolution over time."""
        if not HAS_MATPLOTLIB:
            return
        
        metrics_history = results.get('metrics_history', [])
        if not metrics_history:
            return
        
        epochs = list(range(len(metrics_history)))
        
        # Extract parameters
        connectivity = [m.get('connectivity', 0) for m in metrics_history]
        phi = [m.get('phi', 0) for m in metrics_history]
        complexity = [m.get('complexity', 0) for m in metrics_history]
        qcm = [m.get('qualia_coherence', 0) for m in metrics_history]
        score = [m.get('consciousness_score', 0) for m in metrics_history]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Consciousness Parameter Evolution', fontsize=16, fontweight='bold')
        
        # Connectivity
        ax = axes[0, 0]
        ax.plot(epochs, connectivity, 'b-', linewidth=2, label='Connectivity')
        ax.axhline(y=15.0, color='r', linestyle='--', label='Threshold (15.0)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('⟨k⟩')
        ax.set_title('Connectivity Degree')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Phi
        ax = axes[0, 1]
        ax.plot(epochs, phi, 'g-', linewidth=2, label='Φ')
        ax.axhline(y=0.65, color='r', linestyle='--', label='Threshold (0.65)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Φ')
        ax.set_title('Information Integration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Complexity
        ax = axes[1, 0]
        ax.plot(epochs, complexity, 'm-', linewidth=2, label='Complexity')
        ax.axhline(y=0.8, color='r', linestyle='--', label='Threshold (0.8)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('C')
        ax.set_title('Dynamic Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # QCM
        ax = axes[1, 1]
        ax.plot(epochs, qcm, 'c-', linewidth=2, label='QCM')
        ax.axhline(y=0.75, color='r', linestyle='--', label='Threshold (0.75)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('QCM')
        ax.set_title('Qualia Coherence Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Consciousness Score
        ax = axes[2, 0]
        ax.plot(epochs, score, 'k-', linewidth=2, label='Score')
        ax.axhline(y=1.0, color='r', linestyle='--', label='Critical (1.0)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Consciousness Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # All parameters normalized
        ax = axes[2, 1]
        thresholds = {'Connectivity': 15.0, 'Φ': 0.65, 'Complexity': 0.8, 'QCM': 0.75}
        normalized_connectivity = [c / 15.0 for c in connectivity]
        normalized_phi = [p / 0.65 for p in phi]
        normalized_complexity = [c / 0.8 for c in complexity]
        normalized_qcm = [q / 0.75 for q in qcm]
        
        ax.plot(epochs, normalized_connectivity, 'b-', linewidth=1.5, label='Connectivity/15', alpha=0.7)
        ax.plot(epochs, normalized_phi, 'g-', linewidth=1.5, label='Φ/0.65', alpha=0.7)
        ax.plot(epochs, normalized_complexity, 'm-', linewidth=1.5, label='Complexity/0.8', alpha=0.7)
        ax.plot(epochs, normalized_qcm, 'c-', linewidth=1.5, label='QCM/0.75', alpha=0.7)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Critical')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalized Value')
        ax.set_title('All Parameters (Normalized)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename is None:
            filename = "consciousness_evolution.png"
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Generated: {filepath}")
    
    def plot_performance_benchmarks(self, results: Dict, filename: str = None):
        """Plot performance benchmark results."""
        if not HAS_MATPLOTLIB:
            return
        
        # Evolution speed
        if 'evolution_speed' in results:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('System Performance Benchmarks', fontsize=16, fontweight='bold')
            
            configs = results['evolution_speed'].get('configurations', [])
            if configs:
                neurons = [c.get('neurons', 0) for c in configs]
                time_per_step = [c.get('time_per_step', 0) * 1000 for c in configs]
                throughput = [c.get('neurons_per_second', 0) / 1e6 for c in configs]
                
                # Time per step
                ax = axes[0]
                ax.plot(neurons, time_per_step, 'bo-', linewidth=2, markersize=8)
                ax.set_xlabel('Number of Neurons')
                ax.set_ylabel('Time per Step (ms)')
                ax.set_title('Evolution Speed')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                
                # Throughput
                ax = axes[1]
                ax.plot(neurons, throughput, 'go-', linewidth=2, markersize=8)
                ax.set_xlabel('Number of Neurons')
                ax.set_ylabel('Throughput (M neurons/s)')
                ax.set_title('Processing Throughput')
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if filename is None:
                filename = "performance_benchmarks.png"
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Generated: {filepath}")
    
    def plot_comparative_benchmarks(self, results: Dict, filename: str = None):
        """Plot comparative benchmark results."""
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Comparative Benchmarks: CHIMERA vs PyTorch', fontsize=16, fontweight='bold')
        
        # Matrix operations
        if 'matrix_operations' in results:
            mat = results['matrix_operations']
            ax = axes[0]
            
            labels = []
            times = []
            
            if 'pytorch' in mat and mat['pytorch']:
                labels.append('PyTorch')
                times.append(mat['pytorch'].get('time_per_op', 0) * 1000)
            
            labels.append('CHIMERA')
            times.append(mat.get('chimera', {}).get('time_per_op', 0) * 1000)
            
            if labels and times:
                colors = ['#FF6B6B', '#4ECDC4']
                bars = ax.bar(labels, times, color=colors[:len(labels)])
                ax.set_ylabel('Time per Operation (ms)')
                ax.set_title('Matrix Operations')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}ms', ha='center', va='bottom')
        
        # Memory footprint
        if 'memory_footprint' in results:
            mem = results['memory_footprint']
            ax = axes[1]
            
            labels = []
            memories = []
            
            if 'pytorch' in mem and mem['pytorch']:
                labels.append('PyTorch')
                memories.append(mem['pytorch'].get('memory_mb', 0))
            
            labels.append('CHIMERA')
            memories.append(mem.get('chimera', {}).get('memory_mb', 0))
            
            if labels and memories:
                colors = ['#FF6B6B', '#4ECDC4']
                bars = ax.bar(labels, memories, color=colors[:len(labels)])
                ax.set_ylabel('Memory Usage (MB)')
                ax.set_title('Memory Footprint')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}MB', ha='center', va='bottom')
        
        # Synaptic updates
        if 'synaptic_updates' in results:
            syn = results['synaptic_updates']
            ax = axes[2]
            
            labels = []
            times = []
            
            if 'pytorch' in syn and syn['pytorch']:
                labels.append('PyTorch')
                times.append(syn['pytorch'].get('time_per_update', 0) * 1000)
            
            labels.append('CHIMERA')
            times.append(syn.get('chimera', {}).get('time_per_update', 0) * 1000)
            
            if labels and times:
                colors = ['#FF6B6B', '#4ECDC4']
                bars = ax.bar(labels, times, color=colors[:len(labels)])
                ax.set_ylabel('Time per Update (ms)')
                ax.set_title('Synaptic Updates')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if filename is None:
            filename = "comparative_benchmarks.png"
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Generated: {filepath}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available. Skipping visualizations.")
            return
        
        # Consciousness evolution plots
        for filepath in self.results_dir.glob("consciousness_emergence_*.json"):
            with open(filepath, 'r') as f:
                results = json.load(f)
                filename = filepath.stem + "_evolution.png"
                self.plot_consciousness_evolution(results, filename)
        
        # Performance benchmarks
        perf_file = Path(__file__).parent / "benchmarks" / "system_benchmark_results.json"
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                results = json.load(f)
                self.plot_performance_benchmarks(results)
        
        # Comparative benchmarks
        comp_file = Path(__file__).parent / "benchmarks" / "comparative_benchmark_results.json"
        if comp_file.exists():
            with open(comp_file, 'r') as f:
                results = json.load(f)
                self.plot_comparative_benchmarks(results)
        
        print("\n[OK] All visualizations generated")


def main():
    """Main execution."""
    generator = VisualizationGenerator()
    generator.generate_all_visualizations()


if __name__ == '__main__':
    main()

