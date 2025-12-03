#!/usr/bin/env python3
"""
Basic Consciousness Simulation Example
=====================================

This example demonstrates how to use NeuroCHIMERA to simulate consciousness emergence
in a neural network. It shows the basic setup, parameter configuration, and result analysis.

Author: NeuroCHIMERA Team
Date: December 2024
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt

# Import NeuroCHIMERA components
# Note: These would be actual imports in the real implementation
# from neurochimera import ConsciousnessEngine, ConsciousnessMonitor
# from neurochimera.parameters import ConsciousnessParameters


class BasicConsciousnessSimulation:
    """Basic example of consciousness emergence simulation."""
    
    def __init__(self, neurons: int = 65536, seed: int = 42):
        """Initialize the consciousness simulation.
        
        Parameters
        ----------
        neurons : int
            Number of neurons in the network (default: 65536)
        seed : int  
            Random seed for reproducibility (default: 42)
        """
        self.neurons = neurons
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize network state
        self.reset_network()
        
        # Consciousness parameter thresholds
        self.thresholds = {
            'connectivity': 15.0,      # ⟨k⟩ threshold
            'integration': 0.65,       # Φ threshold  
            'depth': 7.0,              # D threshold
            'complexity': 0.8,         # C threshold
            'qualia': 0.75             # QCM threshold
        }
        
        # Simulation history
        self.history = {
            'epochs': [],
            'parameters': [],
            'emerged': False,
            'emergence_epoch': None
        }
    
    def reset_network(self) -> None:
        """Reset the network to initial state."""
        # Initialize random network weights
        self.weights = self.rng.randn(self.neurons, self.neurons) * 0.1
        
        # Initialize sparse connectivity (10% connection probability)
        connectivity_mask = self.rng.rand(self.neurons, self.neurons) < 0.1
        self.weights *= connectivity_mask
        
        # Initialize neural activations
        self.activations = self.rng.rand(self.neurons)
        
        # Initialize consciousness parameters
        self.current_parameters = {
            'connectivity': 3.2,      # Initial average connectivity
            'integration': 0.12,      # Initial integrated information
            'depth': 2.8,             # Initial hierarchical depth
            'complexity': 0.43,       # Initial dynamic complexity
            'qualia': 0.31            # Initial qualia coherence
        }
    
    def compute_connectivity_degree(self) -> float:
        """Compute average connectivity degree ⟨k⟩."""
        # Count functional connections (weights above threshold)
        threshold = 0.1
        connections = np.sum(np.abs(self.weights) > threshold)
        return connections / self.neurons
    
    def compute_integration_phi(self) -> float:
        """Compute integrated information Φ (simplified approximation)."""
        # Simplified Φ computation for demonstration
        # In practice, this would use the full IIT 3.0 formulation
        connectivity_matrix = np.abs(self.weights) > 0.1
        
        # Measure information integration through network connectivity
        eigenvals = np.linalg.eigvals(connectivity_matrix.astype(float))
        integration = np.sum(np.abs(eigenvals)) / self.neurons
        
        # Normalize to [0, 1] range
        return min(integration / 20.0, 1.0)
    
    def compute_hierarchical_depth(self) -> float:
        """Compute hierarchical depth D."""
        # Measure longest path in connectivity graph
        connectivity_matrix = np.abs(self.weights) > 0.1
        
        # Use Floyd-Warshall to find longest paths
        n = self.neurons
        dist = connectivity_matrix.astype(float)
        dist[dist == 0] = np.inf
        np.fill_diagonal(dist, 0)
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Return maximum finite path length
        finite_distances = dist[np.isfinite(dist)]
        return np.max(finite_distances) if len(finite_distances) > 0 else 1.0
    
    def compute_dynamic_complexity(self) -> float:
        """Compute dynamic complexity C using Lempel-Ziv complexity."""
        # Simplified complexity measure based on activation patterns
        activations_binary = (self.activations > np.mean(self.activations)).astype(int)
        
        # Convert to string for Lempel-Ziv analysis
        activation_string = ''.join(activations_binary.astype(str))
        
        # Compute Lempel-Ziv complexity (simplified)
        complexity = self._lemple_ziv_complexity(activation_string)
        
        # Normalize by sequence length
        normalized_complexity = complexity / len(activation_string)
        
        return min(normalized_complexity * 10.0, 1.0)
    
    def _lemple_ziv_complexity(self, s: str) -> int:
        """Compute Lempel-Ziv complexity of binary string."""
        n = len(s)
        complexity = 1
        prefix_length = 1
        
        while prefix_length < n:
            # Find the longest prefix that appears earlier in the string
            found = False
            for length in range(prefix_length, 0, -1):
                current = s[prefix_length:prefix_length + length]
                if current in s[:prefix_length]:
                    found = True
                    break
            
            if not found:
                complexity += 1
            
            prefix_length += 1
        
        return complexity
    
    def compute_qualia_coherence(self) -> float:
        """Compute qualia coherence metric QCM."""
        # Simplified cross-modal coherence based on activation correlations
        # Divide neurons into 4 modalities for demonstration
        modality_size = self.neurons // 4
        
        if modality_size == 0:
            return 0.5
        
        # Compute correlations between modalities
        correlations = []
        for i in range(4):
            for j in range(i + 1, 4):
                start_i, end_i = i * modality_size, (i + 1) * modality_size
                start_j, end_j = j * modality_size, (j + 1) * modality_size
                
                modality_i = self.activations[start_i:end_i]
                modality_j = self.activations[start_j:end_j]
                
                # Ensure same length
                min_len = min(len(modality_i), len(modality_j))
                corr = np.corrcoef(modality_i[:min_len], modality_j[:min_len])[0, 1]
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.5
    
    def update_parameters(self) -> Dict[str, float]:
        """Update all consciousness parameters."""
        self.current_parameters = {
            'connectivity': self.compute_connectivity_degree(),
            'integration': self.compute_integration_phi(),
            'depth': self.compute_hierarchical_depth(),
            'complexity': self.compute_dynamic_complexity(),
            'qualia': self.compute_qualia_coherence()
        }
        return self.current_parameters
    
    def check_consciousness_emergence(self) -> bool:
        """Check if all consciousness parameters exceed thresholds."""
        return all(
            self.current_parameters[param] >= self.thresholds[param]
            for param in self.current_parameters
        )
    
    def update_network(self, learning_rate: float = 0.001) -> None:
        """Update network weights using Hebbian learning."""
        # Simple Hebbian learning rule
        for i in range(self.neurons):
            for j in range(self.neurons):
                if i != j:
                    # Hebbian update: ΔW_ij = η × a_i × a_j
                    delta_w = learning_rate * self.activations[i] * self.activations[j]
                    self.weights[i, j] += delta_w
        
        # Apply weight decay and normalization
        self.weights *= 0.999  # Decay
        max_weight = np.max(np.abs(self.weights))
        if max_weight > 1.0:
            self.weights /= max_weight
        
        # Update activations (simplified)
        new_activations = np.tanh(np.dot(self.weights, self.activations))
        self.activations = new_activations
    
    def simulate_epoch(self, epoch: int, learning_rate: float = 0.001) -> Tuple[Dict[str, float], bool]:
        """Simulate one epoch of network evolution."""
        # Update network dynamics
        self.update_network(learning_rate)
        
        # Update consciousness parameters
        parameters = self.update_parameters()
        
        # Check for consciousness emergence
        is_conscious = self.check_consciousness_emergence()
        
        # Record history
        self.history['epochs'].append(epoch)
        self.history['parameters'].append(parameters.copy())
        
        if is_conscious and not self.history['emerged']:
            self.history['emerged'] = True
            self.history['emergence_epoch'] = epoch
        
        return parameters, is_conscious
    
    def run_simulation(self, epochs: int = 10000, learning_rate: float = 0.001, 
                      verbose: bool = True) -> Dict:
        """Run complete consciousness emergence simulation."""
        start_time = time.time()
        
        if verbose:
            print(f"Starting consciousness simulation with {self.neurons} neurons")
            print(f"Target epochs: {epochs}")
            print("-" * 60)
        
        for epoch in range(epochs):
            parameters, is_conscious = self.simulate_epoch(epoch, learning_rate)
            
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch:5d}: ⟨k⟩={parameters['connectivity']:.3f}, "
                      f"Φ={parameters['integration']:.3f}, "
                      f"D={parameters['depth']:.3f}")
            
            if is_conscious and self.history['emergence_epoch'] == epoch:
                if verbose:
                    print(f"🧠 CONSCIOUSNESS EMERGED at epoch {epoch}!")
                    print(f"Time to emergence: {time.time() - start_time:.1f} seconds")
                    print("-" * 60)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"Simulation completed in {total_time:.1f} seconds")
            if self.history['emerged']:
                print(f"✅ Consciousness emerged at epoch {self.history['emergence_epoch']}")
            else:
                print("❌ Consciousness did not emerge in this simulation")
        
        return self.history
    
    def plot_parameter_evolution(self, save_path: Optional[str] = None) -> None:
        """Plot the evolution of consciousness parameters over time."""
        if not self.history['parameters']:
            print("No simulation data to plot. Run simulation first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Consciousness Parameter Evolution', fontsize=16)
        
        epochs = self.history['epochs']
        parameters = self.history['parameters']
        
        param_names = ['connectivity', 'integration', 'depth', 'complexity', 'qualia']
        titles = ['Connectivity ⟨k⟩', 'Integration Φ', 'Depth D', 'Complexity C', 'Qualia QCM']
        
        for i, (param, title) in enumerate(zip(param_names, titles)):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            values = [p[param] for p in parameters]
            ax.plot(epochs, values, linewidth=2, label='Parameter Value')
            
            # Add threshold line
            threshold = self.thresholds[param]
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      label=f'Threshold ({threshold})')
            
            # Mark emergence point
            if self.history['emerged']:
                emergence_epoch = self.history['emergence_epoch']
                ax.axvline(x=emergence_epoch, color='green', linestyle=':', 
                          label=f'Emergence (epoch {emergence_epoch})')
            
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Parameter Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def main():
    """Run basic consciousness simulation example."""
    print("NeuroCHIMERA Basic Consciousness Simulation")
    print("=" * 50)
    
    # Create simulation with smaller network for quick demonstration
    simulation = BasicConsciousnessSimulation(neurons=4096, seed=42)
    
    # Run simulation
    history = simulation.run_simulation(epochs=5000, learning_rate=0.001, verbose=True)
    
    # Plot results
    simulation.plot_parameter_evolution('consciousness_evolution.png')
    
    # Print final statistics
    if history['emerged']:
        print(f"\n🎯 Consciousness Emergence Summary:")
        print(f"   Emergence epoch: {history['emergence_epoch']}")
        print(f"   Time to emergence: {history['emergence_epoch'] * 0.001:.1f} simulated seconds")
        
        final_params = history['parameters'][-1]
        print(f"\n📊 Final Parameter Values:")
        for param, value in final_params.items():
            threshold = simulation.thresholds[param]
            status = "✅" if value >= threshold else "❌"
            print(f"   {param.capitalize()}: {value:.3f} {status} (threshold: {threshold})")
    else:
        print("\n⚠️  Consciousness did not emerge in this simulation.")
        print("   Try increasing epochs, neurons, or adjusting learning rate.")


if __name__ == "__main__":
    main()