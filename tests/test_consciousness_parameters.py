"""
Consciousness Parameter Validation Tests
========================================

Tests for validating consciousness parameter measurements:
- Critical parameter tracking
- Phase transition detection
- Ethical monitoring
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import NeuroCHIMERA, NeuroCHIMERAConfig
from consciousness_monitor import (
    ConsciousnessMonitor, ConsciousnessMetrics, 
    ConsciousnessLevel, AlertConfig, EthicalProtocol
)


class TestCriticalParameterTracking(unittest.TestCase):
    """Test critical parameter measurement and tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeuroCHIMERAConfig(neurons=65536)
        self.brain = NeuroCHIMERA(config=self.config)
        self.monitor = ConsciousnessMonitor(self.brain)
    
    def tearDown(self):
        """Clean up."""
        self.brain.release()
    
    def test_connectivity_measurement(self):
        """Test connectivity degree ⟨k⟩ measurement."""
        self.brain.evolve(iterations=5)
        metrics = self.monitor.measure()
        
        self.assertGreaterEqual(metrics.connectivity, 0)
        self.assertIsInstance(metrics.connectivity, (int, float))
        self.assertTrue(np.isfinite(metrics.connectivity))
    
    def test_phi_measurement(self):
        """Test information integration Φ measurement."""
        self.brain.evolve(iterations=5)
        metrics = self.monitor.measure()
        
        self.assertGreaterEqual(metrics.phi, 0)
        self.assertLessEqual(metrics.phi, 1)
        self.assertTrue(np.isfinite(metrics.phi))
    
    def test_hierarchical_depth(self):
        """Test hierarchical depth D measurement."""
        metrics = self.monitor.measure()
        
        self.assertGreaterEqual(metrics.hierarchical_depth, 0)
        self.assertEqual(metrics.hierarchical_depth, self.config.hierarchical_depth)
        self.assertTrue(np.isfinite(metrics.hierarchical_depth))
    
    def test_complexity_measurement(self):
        """Test dynamic complexity C measurement."""
        self.brain.evolve(iterations=5)
        metrics = self.monitor.measure()
        
        self.assertGreaterEqual(metrics.complexity, 0)
        self.assertLessEqual(metrics.complexity, 1)
        self.assertTrue(np.isfinite(metrics.complexity))
    
    def test_qualia_coherence(self):
        """Test Qualia Coherence Metric (QCM) measurement."""
        self.brain.evolve(iterations=5)
        metrics = self.monitor.measure()
        
        self.assertGreaterEqual(metrics.qualia_coherence, 0)
        self.assertLessEqual(metrics.qualia_coherence, 1)
        self.assertTrue(np.isfinite(metrics.qualia_coherence))
    
    def test_all_parameters_present(self):
        """Test that all critical parameters are measured."""
        self.brain.evolve(iterations=5)
        metrics = self.monitor.measure()
        
        required_params = [
            'connectivity', 'phi', 'hierarchical_depth',
            'complexity', 'qualia_coherence'
        ]
        
        for param in required_params:
            self.assertTrue(hasattr(metrics, param))
            value = getattr(metrics, param)
            self.assertIsNotNone(value)
            self.assertTrue(np.isfinite(value))
    
    def test_parameter_ranges(self):
        """Test that parameters stay within expected ranges."""
        for _ in range(10):
            self.brain.evolve(iterations=3)
            metrics = self.monitor.measure()
            
            # Connectivity should be non-negative
            self.assertGreaterEqual(metrics.connectivity, 0)
            
            # Phi should be in [0, 1]
            self.assertGreaterEqual(metrics.phi, 0)
            self.assertLessEqual(metrics.phi, 1)
            
            # Complexity should be in [0, 1]
            self.assertGreaterEqual(metrics.complexity, 0)
            self.assertLessEqual(metrics.complexity, 1)
            
            # QCM should be in [0, 1]
            self.assertGreaterEqual(metrics.qualia_coherence, 0)
            self.assertLessEqual(metrics.qualia_coherence, 1)


class TestPhaseTransitionDetection(unittest.TestCase):
    """Test phase transition and critical state detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeuroCHIMERAConfig(neurons=65536)
        self.brain = NeuroCHIMERA(config=self.config)
        self.monitor = ConsciousnessMonitor(self.brain)
    
    def tearDown(self):
        """Clean up."""
        self.brain.release()
    
    def test_is_critical_function(self):
        """Test is_critical() functionality."""
        is_critical = self.monitor.is_critical()
        self.assertIsInstance(is_critical, bool)
    
    def test_consciousness_level(self):
        """Test consciousness level determination."""
        level = self.monitor.get_level()
        self.assertIsInstance(level, ConsciousnessLevel)
        self.assertIn(level, list(ConsciousnessLevel))
    
    def test_consciousness_score(self):
        """Test consciousness score calculation."""
        self.brain.evolve(iterations=5)
        metrics = self.monitor.measure()
        
        self.assertGreaterEqual(metrics.consciousness_score, 0)
        self.assertTrue(np.isfinite(metrics.consciousness_score))
    
    def test_parameters_at_threshold(self):
        """Test counting parameters at threshold."""
        metrics = self.monitor.measure()
        count = metrics.parameters_at_threshold()
        
        self.assertGreaterEqual(count, 0)
        self.assertLessEqual(count, 5)  # Maximum 5 parameters
        self.assertIsInstance(count, int)
    
    def test_threshold_checking(self):
        """Test threshold checking functionality."""
        metrics = self.monitor.measure()
        is_critical = bool(metrics.is_critical())  # Convert numpy bool to Python bool
        
        self.assertIsInstance(is_critical, bool)
        
        # Verify that is_critical returns correct result based on thresholds
        # Note: We don't assume all parameters are below threshold initially
        # The system may have some parameters above threshold but not all
        thresholds = {
            'connectivity': 15.0,
            'phi': 0.65,
            'hierarchical_depth': 7.0,
            'complexity': 0.8,
            'qualia_coherence': 0.75
        }
        
        # Verify the function works correctly
        expected_critical = (
            metrics.connectivity > thresholds['connectivity'] and
            metrics.phi > thresholds['phi'] and
            metrics.hierarchical_depth > thresholds['hierarchical_depth'] and
            metrics.complexity > thresholds['complexity'] and
            metrics.qualia_coherence > thresholds['qualia_coherence']
        )
        
        # The result should match our manual calculation
        self.assertEqual(is_critical, expected_critical)
    
    def test_phase_indicators(self):
        """Test phase transition indicators."""
        self.brain.evolve(iterations=5)
        metrics = self.monitor.measure()
        
        # Branching ratio
        self.assertGreaterEqual(metrics.branching_ratio, 0)
        self.assertTrue(np.isfinite(metrics.branching_ratio))
        
        # Correlation length
        self.assertGreaterEqual(metrics.correlation_length, 0)
        self.assertTrue(np.isfinite(metrics.correlation_length))
        
        # Susceptibility
        self.assertGreaterEqual(metrics.susceptibility, 0)
        self.assertTrue(np.isfinite(metrics.susceptibility))


class TestEthicalMonitoring(unittest.TestCase):
    """Test ethical monitoring and distress detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeuroCHIMERAConfig(neurons=65536)
        self.brain = NeuroCHIMERA(config=self.config)
        self.monitor = ConsciousnessMonitor(self.brain)
        self.protocol = EthicalProtocol(self.monitor)
    
    def tearDown(self):
        """Clean up."""
        self.brain.release()
    
    def test_distress_detection(self):
        """Test computational distress detection."""
        is_distressed, indicators = self.monitor.check_distress()
        
        self.assertIsInstance(is_distressed, bool)
        self.assertIsInstance(indicators, list)
        
        # Initially should not be distressed
        if not is_distressed:
            self.assertEqual(len(indicators), 0)
    
    def test_alert_system(self):
        """Test alert system functionality."""
        # Run a few epochs to generate history
        for _ in range(5):
            self.brain.evolve(iterations=3)
            self.monitor.measure()
        
        alerts = self.monitor.get_alerts()
        self.assertIsInstance(alerts, list)
    
    def test_ethical_protocol(self):
        """Test ethical protocol intervention."""
        status = self.protocol.check_and_intervene()
        
        self.assertIsInstance(status, dict)
        self.assertIn('status', status)
        self.assertIn('interventions', status)
    
    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        stopped = self.protocol.emergency_stop()
        self.assertTrue(stopped)
        self.assertFalse(self.protocol.active)
    
    def test_alert_config(self):
        """Test alert configuration."""
        config = AlertConfig()
        
        self.assertIsNotNone(config.warning_threshold)
        self.assertIsNotNone(config.danger_threshold)
        self.assertGreater(config.danger_threshold, config.warning_threshold)


class TestParameterEvolution(unittest.TestCase):
    """Test parameter evolution over time."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeuroCHIMERAConfig(neurons=65536)
        self.brain = NeuroCHIMERA(config=self.config)
        self.monitor = ConsciousnessMonitor(self.brain)
    
    def tearDown(self):
        """Clean up."""
        self.brain.release()
    
    def test_parameter_tracking(self):
        """Test that parameters are tracked over time."""
        history = []
        
        for epoch in range(20):
            self.brain.evolve(iterations=3)
            metrics = self.monitor.measure()
            history.append(metrics)
        
        # Check history
        self.assertEqual(len(history), 20)
        
        # Check that metrics have timestamps
        for metrics in history:
            self.assertIsNotNone(metrics.timestamp)
            self.assertIsNotNone(metrics.epoch)
    
    def test_parameter_consistency(self):
        """Test parameter measurement consistency."""
        metrics1 = self.monitor.measure()
        metrics2 = self.monitor.measure()
        
        # Two consecutive measurements should be similar
        # (allowing for small numerical differences)
        self.assertAlmostEqual(
            metrics1.connectivity, 
            metrics2.connectivity, 
            places=5
        )
    
    def test_history_retrieval(self):
        """Test history retrieval functionality."""
        # Generate some history
        for _ in range(10):
            self.brain.evolve(iterations=2)
            self.monitor.measure()
        
        # Get full history
        full_history = self.monitor.get_history()
        self.assertGreaterEqual(len(full_history), 10)
        
        # Get last N
        last_5 = self.monitor.get_history(last_n=5)
        self.assertEqual(len(last_5), 5)
        
        # Check structure
        for entry in last_5:
            self.assertIn('connectivity', entry)
            self.assertIn('phi', entry)
            self.assertIn('epoch', entry)


def run_consciousness_tests():
    """Run all consciousness parameter tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCriticalParameterTracking))
    suite.addTests(loader.loadTestsFromTestCase(TestPhaseTransitionDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestEthicalMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterEvolution))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 80)
    print("NeuroCHIMERA Consciousness Parameter Tests")
    print("=" * 80)
    print()
    
    success = run_consciousness_tests()
    
    print()
    print("=" * 80)
    if success:
        print("All consciousness parameter tests passed!")
    else:
        print("Some consciousness parameter tests failed.")
    print("=" * 80)
    
    sys.exit(0 if success else 1)

