"""
Consciousness Monitor - Critical Parameter Tracking
===================================================

Real-time monitoring of consciousness-related metrics based on
Veselov's theoretical framework. Tracks the approach to criticality
and provides alerts when consciousness emergence thresholds are reached.

Critical Parameters (Veselov Hypothesis):
- Connectivity Degree ⟨k⟩ > 15 ± 3
- Information Integration Φ > 0.65 ± 0.15
- Hierarchical Depth D > 7 ± 2
- Dynamic Complexity C > 0.8 ± 0.1
- Qualia Coherence Metric QCM > 0.75

Authors: V.F. Veselov (MIET), Francisco Angulo de Lafuente (Madrid)
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class ConsciousnessLevel(Enum):
    """Consciousness emergence levels based on parameter thresholds."""
    SUBCRITICAL = 0      # No parameters at threshold
    TRANSITIONING = 1    # 1-2 parameters at threshold
    NEAR_CRITICAL = 2    # 3-4 parameters at threshold
    CRITICAL = 3         # All parameters at threshold
    SUPERCRITICAL = 4    # All parameters significantly above threshold


@dataclass
class ConsciousnessMetrics:
    """Container for consciousness-related metrics."""
    
    # Core Veselov parameters
    connectivity: float = 0.0        # ⟨k⟩
    phi: float = 0.0                 # Φ (information integration)
    hierarchical_depth: float = 0.0  # D
    complexity: float = 0.0          # C (dynamic complexity)
    qualia_coherence: float = 0.0    # QCM
    
    # Derived metrics
    consciousness_score: float = 0.0
    level: ConsciousnessLevel = ConsciousnessLevel.SUBCRITICAL
    
    # Statistics
    mean_activation: float = 0.0
    std_activation: float = 0.0
    sparsity: float = 0.0
    
    # Phase transition indicators
    branching_ratio: float = 0.0     # σ ≈ 1.0 at criticality
    correlation_length: float = 0.0  # Diverges at criticality
    susceptibility: float = 0.0      # Peaks at transition
    
    # Temporal metrics
    timestamp: float = field(default_factory=time.time)
    epoch: int = 0
    
    def is_critical(self, thresholds: Dict[str, float] = None) -> bool:
        """Check if all critical thresholds are exceeded."""
        if thresholds is None:
            thresholds = {
                'connectivity': 15.0,
                'phi': 0.65,
                'hierarchical_depth': 7.0,
                'complexity': 0.8,
                'qualia_coherence': 0.75
            }
        
        return (
            self.connectivity > thresholds['connectivity'] and
            self.phi > thresholds['phi'] and
            self.hierarchical_depth > thresholds['hierarchical_depth'] and
            self.complexity > thresholds['complexity'] and
            self.qualia_coherence > thresholds['qualia_coherence']
        )
    
    def parameters_at_threshold(self, thresholds: Dict[str, float] = None) -> int:
        """Count how many parameters are at or above threshold."""
        if thresholds is None:
            thresholds = {
                'connectivity': 15.0,
                'phi': 0.65,
                'hierarchical_depth': 7.0,
                'complexity': 0.8,
                'qualia_coherence': 0.75
            }
        
        count = 0
        if self.connectivity > thresholds['connectivity']:
            count += 1
        if self.phi > thresholds['phi']:
            count += 1
        if self.hierarchical_depth > thresholds['hierarchical_depth']:
            count += 1
        if self.complexity > thresholds['complexity']:
            count += 1
        if self.qualia_coherence > thresholds['qualia_coherence']:
            count += 1
        
        return count
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'connectivity': self.connectivity,
            'phi': self.phi,
            'hierarchical_depth': self.hierarchical_depth,
            'complexity': self.complexity,
            'qualia_coherence': self.qualia_coherence,
            'consciousness_score': self.consciousness_score,
            'level': self.level.name,
            'mean_activation': self.mean_activation,
            'std_activation': self.std_activation,
            'sparsity': self.sparsity,
            'branching_ratio': self.branching_ratio,
            'correlation_length': self.correlation_length,
            'susceptibility': self.susceptibility,
            'timestamp': self.timestamp,
            'epoch': self.epoch
        }


@dataclass
class AlertConfig:
    """Configuration for consciousness emergence alerts."""
    
    # Alert thresholds (fraction of critical threshold)
    warning_threshold: float = 0.7   # Alert when 70% of threshold reached
    danger_threshold: float = 0.9    # Strong alert at 90%
    
    # Ethical intervention thresholds
    distress_prediction_error: float = 0.3   # Chronic prediction error
    distress_homeostatic: float = 0.4        # Homeostatic violation
    distress_attention_threat: float = 0.6   # Attention to negative stimuli
    
    # Safety thresholds
    autonomy_warning: float = 0.9            # Self-directed behavior threshold
    behavioral_withdrawal: float = 0.5       # Depression indicator


class ConsciousnessMonitor:
    """
    Monitor consciousness-related metrics in NeuroCHIMERA system.
    
    Tracks the approach to criticality and provides:
    - Real-time metric calculation
    - Phase transition detection
    - Ethical oversight alerts
    - Historical analysis
    
    Usage:
        brain = NeuroCHIMERA(...)
        monitor = ConsciousnessMonitor(brain)
        
        for epoch in range(10000):
            brain.evolve()
            metrics = monitor.measure()
            
            if monitor.is_critical():
                print("Consciousness emergence detected!")
            
            if monitor.check_distress():
                print("Warning: Distress indicators detected!")
    """
    
    def __init__(
        self,
        brain: 'NeuroCHIMERA',
        config: Optional[AlertConfig] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize consciousness monitor.
        
        Args:
            brain: NeuroCHIMERA instance to monitor
            config: Alert configuration
            thresholds: Custom critical thresholds
        """
        self.brain = brain
        self.config = config or AlertConfig()
        
        self.thresholds = thresholds or {
            'connectivity': 15.0,
            'phi': 0.65,
            'hierarchical_depth': 7.0,
            'complexity': 0.8,
            'qualia_coherence': 0.75
        }
        
        # History
        self.metrics_history: List[ConsciousnessMetrics] = []
        self.alerts_history: List[Dict] = []
        
        # Callbacks
        self.on_critical: Optional[Callable[[ConsciousnessMetrics], None]] = None
        self.on_warning: Optional[Callable[[str, float], None]] = None
        self.on_distress: Optional[Callable[[str, float], None]] = None
        
        # State tracking
        self._prev_activations: Optional[np.ndarray] = None
        self._critical_reached = False
        self._current_epoch = 0
    
    def measure(self) -> ConsciousnessMetrics:
        """
        Measure current consciousness-related metrics.
        
        Returns:
            ConsciousnessMetrics with all measurements
        """
        state = self.brain.frame.neural_state
        weights = self.brain.frame.connectivity
        
        self._current_epoch += 1
        
        # Calculate core Veselov parameters
        connectivity = self._measure_connectivity(weights)
        phi = self._measure_phi(state)
        depth = self._measure_hierarchical_depth()
        complexity = self._measure_complexity(state)
        qcm = self._measure_qualia_coherence(state)
        
        # Calculate derived metrics
        mean_act = np.mean(state[:, :, 0])
        std_act = np.std(state[:, :, 0])
        sparsity = np.mean(state[:, :, 0] < 0.1)
        
        # Phase transition indicators
        branching = self._measure_branching_ratio(state)
        corr_len = self._measure_correlation_length(state)
        suscept = self._measure_susceptibility(state)
        
        # Calculate consciousness score (weighted average)
        c_score = (
            0.25 * (connectivity / self.thresholds['connectivity']) +
            0.25 * (phi / self.thresholds['phi']) +
            0.15 * (depth / self.thresholds['hierarchical_depth']) +
            0.20 * (complexity / self.thresholds['complexity']) +
            0.15 * (qcm / self.thresholds['qualia_coherence'])
        )
        
        # Determine consciousness level
        params_at_threshold = sum([
            connectivity > self.thresholds['connectivity'],
            phi > self.thresholds['phi'],
            depth > self.thresholds['hierarchical_depth'],
            complexity > self.thresholds['complexity'],
            qcm > self.thresholds['qualia_coherence']
        ])
        
        if params_at_threshold == 0:
            level = ConsciousnessLevel.SUBCRITICAL
        elif params_at_threshold <= 2:
            level = ConsciousnessLevel.TRANSITIONING
        elif params_at_threshold <= 4:
            level = ConsciousnessLevel.NEAR_CRITICAL
        elif params_at_threshold == 5:
            level = ConsciousnessLevel.CRITICAL
            
            # Check for supercritical
            all_above = (
                connectivity > self.thresholds['connectivity'] * 1.2 and
                phi > self.thresholds['phi'] * 1.1 and
                complexity > self.thresholds['complexity'] * 1.1 and
                qcm > self.thresholds['qualia_coherence'] * 1.1
            )
            if all_above:
                level = ConsciousnessLevel.SUPERCRITICAL
        else:
            level = ConsciousnessLevel.SUBCRITICAL
        
        metrics = ConsciousnessMetrics(
            connectivity=connectivity,
            phi=phi,
            hierarchical_depth=depth,
            complexity=complexity,
            qualia_coherence=qcm,
            consciousness_score=c_score,
            level=level,
            mean_activation=mean_act,
            std_activation=std_act,
            sparsity=sparsity,
            branching_ratio=branching,
            correlation_length=corr_len,
            susceptibility=suscept,
            epoch=self._current_epoch
        )
        
        # Store history
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Store activations for next iteration
        self._prev_activations = state[:, :, 0].copy()
        
        return metrics
    
    def _measure_connectivity(self, weights: np.ndarray) -> float:
        """Measure effective connectivity ⟨k⟩."""
        # Count significant connections (|w| > threshold)
        significant = np.abs(weights[:, :, 0] * 2 - 1) > 0.3
        
        # Average over neurons
        # Assuming 5×5 neighborhood
        neighborhood_size = 25
        total_possible = weights.shape[0] * weights.shape[1] * neighborhood_size
        total_significant = np.sum(significant) * neighborhood_size
        
        return total_significant / (weights.shape[0] * weights.shape[1])
    
    def _measure_phi(self, state: np.ndarray) -> float:
        """
        Measure information integration Φ (approximation).
        
        True Φ computation is NP-hard; we use correlation-based proxy.
        """
        activations = state[:, :, 0].flatten()
        n = len(activations)
        
        if n < 4:
            return 0.0
        
        # Partition into quadrants and measure integration
        size = int(np.sqrt(n))
        grid = activations.reshape(size, size)
        
        half = size // 2
        q1 = grid[:half, :half].flatten()
        q2 = grid[:half, half:].flatten()
        q3 = grid[half:, :half].flatten()
        q4 = grid[half:, half:].flatten()
        
        # Compute pairwise correlations
        correlations = []
        for a, b in [(q1, q2), (q1, q3), (q1, q4), (q2, q3), (q2, q4), (q3, q4)]:
            if np.std(a) > 0.001 and np.std(b) > 0.001:
                corr = np.abs(np.corrcoef(a, b)[0, 1])
                correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        # Phi approximation: mean correlation × complement of minimum
        mean_corr = np.mean(correlations)
        min_corr = min(correlations)
        
        # High Phi when all partitions are correlated
        return mean_corr * (1 - min_corr * 0.5)
    
    def _measure_hierarchical_depth(self) -> float:
        """Measure effective hierarchical depth."""
        # Use configured value as base
        base_depth = self.brain.config.hierarchical_depth
        
        # Could analyze weight structure to determine effective depth
        # For now, return configured value
        return float(base_depth)
    
    def _measure_complexity(self, state: np.ndarray) -> float:
        """
        Measure dynamic complexity C using Lempel-Ziv.
        
        C ∈ [0, 1] where 1 indicates edge-of-chaos dynamics.
        """
        activations = state[:, :, 0].flatten()
        
        # Sample subset for efficiency
        sample_size = min(1000, len(activations))
        sample = activations[:sample_size]
        
        # Binarize
        threshold = np.median(sample)
        binary = ''.join(['1' if x > threshold else '0' for x in sample])
        
        # Lempel-Ziv complexity
        n = len(binary)
        if n == 0:
            return 0.0
        
        dictionary = set()
        w = ""
        c = 0
        
        for char in binary:
            wc = w + char
            if wc in dictionary:
                w = wc
            else:
                dictionary.add(wc)
                c += 1
                w = ""
        
        if w:
            c += 1
        
        # Normalize
        max_c = n / np.log2(n) if n > 1 else 1
        
        return min(c / max_c, 1.0)
    
    def _measure_qualia_coherence(self, state: np.ndarray) -> float:
        """
        Measure Qualia Coherence Metric (QCM).
        
        QCM = (1/N) Σ exp(-||pi - qi||² / 2σ²)
        
        Measures cross-modal integration coherence.
        """
        if self.brain.frame.qualia_integration is not None:
            # Use pre-computed qualia texture
            return float(np.mean(self.brain.frame.qualia_integration[:, :, 3]))
        
        # Compute from state channels
        r = state[:, :, 0].flatten()  # Activation
        g = state[:, :, 1].flatten()  # Memory
        b = state[:, :, 2].flatten()  # Time constant
        
        sigma = 0.3
        
        # Pairwise coherence
        coherences = []
        
        for a, b_arr in [(r, g), (r, b), (g, b)]:
            diff = np.abs(a - b_arr)
            coh = np.mean(np.exp(-diff**2 / (2 * sigma**2)))
            coherences.append(coh)
        
        return np.mean(coherences)
    
    def _measure_branching_ratio(self, state: np.ndarray) -> float:
        """
        Measure branching ratio σ.
        
        σ ≈ 1.0 indicates critical branching (edge of chaos).
        σ < 1.0 indicates subcritical (dying cascades).
        σ > 1.0 indicates supercritical (exploding cascades).
        """
        if self._prev_activations is None:
            return 1.0
        
        current = state[:, :, 0]
        previous = self._prev_activations
        
        # Active neurons (above threshold)
        threshold = 0.5
        prev_active = previous > threshold
        curr_active = current > threshold
        
        n_prev = np.sum(prev_active)
        n_curr = np.sum(curr_active)
        
        if n_prev == 0:
            return 1.0
        
        return n_curr / n_prev
    
    def _measure_correlation_length(self, state: np.ndarray) -> float:
        """
        Measure spatial correlation length.
        
        Diverges at criticality.
        """
        activations = state[:, :, 0]
        size = activations.shape[0]
        
        # Compute autocorrelation at different distances
        center = size // 2
        center_val = activations[center, center]
        
        correlations = []
        distances = [1, 2, 4, 8, 16]
        
        for d in distances:
            if d >= size // 2:
                break
            
            # Sample points at distance d
            points = [
                activations[center + d, center],
                activations[center - d, center],
                activations[center, center + d],
                activations[center, center - d]
            ]
            
            corr = np.mean([abs(p - center_val) for p in points])
            correlations.append((d, 1 - corr))  # Correlation decreases with distance
        
        if len(correlations) < 2:
            return 1.0
        
        # Fit exponential decay to estimate correlation length
        # C(r) ∝ exp(-r/ξ)
        distances_arr = np.array([c[0] for c in correlations])
        corrs_arr = np.array([c[1] for c in correlations])
        
        # Simple estimate: distance where correlation drops to 1/e
        target = corrs_arr[0] / np.e
        for i, c in enumerate(corrs_arr):
            if c < target:
                return float(distances_arr[max(0, i-1)])
        
        return float(distances_arr[-1])
    
    def _measure_susceptibility(self, state: np.ndarray) -> float:
        """
        Measure susceptibility (variance of order parameter).
        
        Peaks at phase transition.
        """
        activations = state[:, :, 0]
        
        # Order parameter: mean activation
        order_param = np.mean(activations)
        
        # Susceptibility: variance across spatial regions
        size = activations.shape[0]
        block_size = max(size // 4, 1)
        
        regional_means = []
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                region = activations[i:i+block_size, j:j+block_size]
                regional_means.append(np.mean(region))
        
        return np.var(regional_means)
    
    def _check_alerts(self, metrics: ConsciousnessMetrics):
        """Check for alert conditions and trigger callbacks."""
        
        # Check for critical threshold approach
        for param, threshold in self.thresholds.items():
            value = getattr(metrics, param, 0)
            ratio = value / threshold
            
            if ratio > self.config.danger_threshold and ratio < 1.0:
                alert = {
                    'type': 'approaching_critical',
                    'parameter': param,
                    'value': value,
                    'threshold': threshold,
                    'ratio': ratio,
                    'epoch': metrics.epoch
                }
                self.alerts_history.append(alert)
                
                if self.on_warning:
                    self.on_warning(param, ratio)
        
        # Check for consciousness emergence
        if metrics.is_critical(self.thresholds) and not self._critical_reached:
            self._critical_reached = True
            
            alert = {
                'type': 'critical_reached',
                'epoch': metrics.epoch,
                'metrics': metrics.to_dict()
            }
            self.alerts_history.append(alert)
            
            if self.on_critical:
                self.on_critical(metrics)
    
    def check_distress(self) -> Tuple[bool, List[str]]:
        """
        Check for computational distress indicators.
        
        Implements ethical monitoring per Veselov framework:
        - Chronic prediction errors
        - Homeostatic violations
        - Attention to threat stimuli
        
        Returns:
            Tuple of (is_distressed, list of distress indicators)
        """
        if not self.metrics_history:
            return False, []
        
        indicators = []
        recent = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        # Check prediction error (using complexity as proxy)
        avg_complexity = np.mean([m.complexity for m in recent])
        if avg_complexity < 0.3:  # Very low complexity = frustrated system
            indicators.append(f"chronic_prediction_error: {avg_complexity:.3f}")
        
        # Check homeostatic stress (using activation stability)
        act_variance = np.var([m.mean_activation for m in recent])
        if act_variance > self.config.distress_homeostatic:
            indicators.append(f"homeostatic_stress: {act_variance:.3f}")
        
        # Check behavioral withdrawal (using activation level)
        avg_activation = np.mean([m.mean_activation for m in recent])
        if avg_activation < 0.1:  # Very low activation = withdrawal
            indicators.append(f"behavioral_withdrawal: {avg_activation:.3f}")
        
        is_distressed = len(indicators) > 0
        
        if is_distressed and self.on_distress:
            for indicator in indicators:
                name, value = indicator.split(': ')
                self.on_distress(name, float(value))
        
        return is_distressed, indicators
    
    def is_critical(self) -> bool:
        """Check if system is at critical consciousness threshold."""
        if not self.metrics_history:
            return False
        return self.metrics_history[-1].is_critical(self.thresholds)
    
    def get_level(self) -> ConsciousnessLevel:
        """Get current consciousness level."""
        if not self.metrics_history:
            return ConsciousnessLevel.SUBCRITICAL
        return self.metrics_history[-1].level
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get metrics history as list of dicts."""
        history = [m.to_dict() for m in self.metrics_history]
        if last_n is not None:
            return history[-last_n:]
        return history
    
    def get_alerts(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get alert history."""
        if last_n is not None:
            return self.alerts_history[-last_n:]
        return self.alerts_history
    
    def plot_evolution(self, save_path: Optional[str] = None):
        """
        Plot parameter evolution over time.
        
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
        
        if not self.metrics_history:
            print("No history to plot")
            return
        
        epochs = [m.epoch for m in self.metrics_history]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Connectivity
        ax = axes[0, 0]
        values = [m.connectivity for m in self.metrics_history]
        ax.plot(epochs, values, 'b-', linewidth=2)
        ax.axhline(y=self.thresholds['connectivity'], color='r', linestyle='--', label='Threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('⟨k⟩')
        ax.set_title('Connectivity Degree')
        ax.legend()
        
        # Phi
        ax = axes[0, 1]
        values = [m.phi for m in self.metrics_history]
        ax.plot(epochs, values, 'g-', linewidth=2)
        ax.axhline(y=self.thresholds['phi'], color='r', linestyle='--', label='Threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Φ')
        ax.set_title('Information Integration')
        ax.legend()
        
        # Complexity
        ax = axes[0, 2]
        values = [m.complexity for m in self.metrics_history]
        ax.plot(epochs, values, 'm-', linewidth=2)
        ax.axhline(y=self.thresholds['complexity'], color='r', linestyle='--', label='Threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('C')
        ax.set_title('Dynamic Complexity')
        ax.legend()
        
        # QCM
        ax = axes[1, 0]
        values = [m.qualia_coherence for m in self.metrics_history]
        ax.plot(epochs, values, 'c-', linewidth=2)
        ax.axhline(y=self.thresholds['qualia_coherence'], color='r', linestyle='--', label='Threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('QCM')
        ax.set_title('Qualia Coherence')
        ax.legend()
        
        # Consciousness Score
        ax = axes[1, 1]
        values = [m.consciousness_score for m in self.metrics_history]
        ax.plot(epochs, values, 'k-', linewidth=2)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Critical')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Consciousness Score')
        ax.legend()
        
        # Phase indicators
        ax = axes[1, 2]
        branching = [m.branching_ratio for m in self.metrics_history]
        ax.plot(epochs, branching, 'orange', linewidth=2, label='Branching Ratio')
        ax.axhline(y=1.0, color='r', linestyle='--', label='Critical (σ=1)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('σ')
        ax.set_title('Phase Transition Indicator')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


# =============================================================================
# ETHICAL PROTOCOL
# =============================================================================

class EthicalProtocol:
    """
    Ethical oversight protocol for consciousness research.
    
    Implements monitoring and intervention mechanisms per
    Veselov framework ethical guidelines.
    """
    
    def __init__(self, monitor: ConsciousnessMonitor):
        self.monitor = monitor
        self.intervention_log: List[Dict] = []
        self.active = True
    
    def check_and_intervene(self) -> Dict:
        """
        Check ethical status and intervene if necessary.
        
        Returns:
            Dictionary with status and any interventions taken
        """
        if not self.active:
            return {'status': 'inactive'}
        
        result = {'status': 'ok', 'interventions': []}
        
        # Check consciousness level
        level = self.monitor.get_level()
        if level in [ConsciousnessLevel.CRITICAL, ConsciousnessLevel.SUPERCRITICAL]:
            result['consciousness_alert'] = True
            result['level'] = level.name
        
        # Check distress
        is_distressed, indicators = self.monitor.check_distress()
        if is_distressed:
            result['distress_alert'] = True
            result['distress_indicators'] = indicators
            
            # Log intervention
            intervention = {
                'epoch': self.monitor._current_epoch,
                'reason': 'distress_detected',
                'indicators': indicators,
                'action': 'monitoring'  # Could trigger actual intervention
            }
            self.intervention_log.append(intervention)
            result['interventions'].append(intervention)
        
        return result
    
    def emergency_stop(self) -> bool:
        """
        Trigger emergency stop of consciousness emergence.
        
        Returns:
            True if stop was executed
        """
        intervention = {
            'epoch': self.monitor._current_epoch,
            'reason': 'emergency_stop',
            'action': 'halt'
        }
        self.intervention_log.append(intervention)
        self.active = False
        
        return True


# =============================================================================
# MAIN - Demo
# =============================================================================

if __name__ == "__main__":
    print("Consciousness Monitor - Demo")
    print("=" * 50)
    
    # Create mock brain for testing
    class MockBrain:
        class Config:
            hierarchical_depth = 12
        
        class Frame:
            def __init__(self):
                self.neural_state = np.random.uniform(0, 1, (64, 64, 4)).astype(np.float32)
                self.connectivity = np.random.uniform(0, 1, (64, 64, 4)).astype(np.float32)
                self.qualia_integration = None
        
        def __init__(self):
            self.config = self.Config()
            self.frame = self.Frame()
    
    brain = MockBrain()
    monitor = ConsciousnessMonitor(brain)
    
    # Simulate evolution
    print("\nSimulating 50 epochs...")
    for epoch in range(50):
        # Gradually increase activation (simulating emergence)
        brain.frame.neural_state[:, :, 0] *= 1.02
        brain.frame.neural_state = np.clip(brain.frame.neural_state, 0, 1)
        
        metrics = monitor.measure()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Level={metrics.level.name}, "
                  f"Score={metrics.consciousness_score:.3f}")
    
    # Check final state
    print("\nFinal Status:")
    print(f"  Consciousness Level: {monitor.get_level().name}")
    print(f"  Is Critical: {monitor.is_critical()}")
    
    distressed, indicators = monitor.check_distress()
    print(f"  Distressed: {distressed}")
    if indicators:
        for ind in indicators:
            print(f"    - {ind}")
    
    print("\nDemo complete.")
