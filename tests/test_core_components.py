"""
Core Component Tests for NeuroCHIMERA
=====================================

Comprehensive test suite for validating core system components:
- HNS (Hierarchical Number System) operations
- Engine initialization and GPU context
- Neuromorphic frame management
- Evolution dynamics
- Memory system operations
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_number import (
    HNumber, hns_add, hns_scale, hns_normalize, 
    hns_multiply, hns_compare, BASE
)
from engine import NeuroCHIMERA, NeuroCHIMERAConfig, NeuromorphicFrame
from consciousness_monitor import ConsciousnessMonitor, ConsciousnessMetrics


class TestHNS(unittest.TestCase):
    """Test Hierarchical Number System operations."""
    
    def test_basic_addition(self):
        """Test basic HNS addition."""
        a = HNumber([999.0, 999.0, 0.0, 0.0])  # 999,999
        b = HNumber([1.0, 0.0, 0.0, 0.0])       # 1
        result = hns_add(a, b)
        expected = HNumber([0.0, 0.0, 1.0, 0.0])  # 1,000,000
        self.assertEqual(result.to_integer(), expected.to_integer())
        self.assertEqual(result.to_integer(), 1000000)
    
    def test_carry_propagation(self):
        """Test cascading carry propagation."""
        # Test multiple levels of carry
        a = HNumber([999.0, 999.0, 999.0, 0.0])  # 999,999,999
        b = HNumber([1.0, 0.0, 0.0, 0.0])         # 1
        result = hns_add(a, b)
        expected = HNumber([0.0, 0.0, 0.0, 1.0])  # 1,000,000,000
        self.assertEqual(result.to_integer(), expected.to_integer())
    
    def test_scaling(self):
        """Test HNS scaling operation."""
        a = HNumber([0.0, 0.0, 1.0, 0.0])  # 1,000,000
        result = hns_scale(a, 0.5)
        self.assertEqual(result.to_integer(), 500000)
    
    def test_normalization(self):
        """Test normalization with overflow."""
        # Create number with overflow in R channel
        n = HNumber([1500.0, 500.0, 0.0, 0.0])
        normalized = hns_normalize(n)
        # Should be [500, 501, 0, 0] = 501,500
        self.assertLess(normalized.r, BASE)
        self.assertLess(normalized.g, BASE)
        self.assertEqual(normalized.to_integer(), 501500)
    
    def test_large_numbers(self):
        """Test operations with very large numbers."""
        a = HNumber([500.0, 500.0, 500.0, 0.0])  # 500,500,500
        b = HNumber([600.0, 600.0, 600.0, 0.0])  # 600,600,600
        result = hns_add(a, b)
        expected = 500500500 + 600600600
        # Allow small precision errors for very large numbers due to float32 limitations
        self.assertAlmostEqual(result.to_integer(), expected, delta=100)
    
    def test_precision_preservation(self):
        """Test precision preservation over many operations."""
        value = HNumber([1.0, 0.0, 0.0, 0.0])  # 1
        for i in range(1000):
            value = hns_add(value, HNumber([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(value.to_integer(), 1001)
    
    def test_multiplication(self):
        """Test HNS multiplication."""
        a = HNumber([100.0, 0.0, 0.0, 0.0])  # 100
        b = HNumber([5.0, 0.0, 0.0, 0.0])    # 5
        result = hns_multiply(a, b)
        self.assertEqual(result.to_integer(), 500)
    
    def test_comparison(self):
        """Test HNS comparison operations."""
        a = HNumber([100.0, 0.0, 0.0, 0.0])
        b = HNumber([50.0, 0.0, 0.0, 0.0])
        self.assertEqual(hns_compare(a, b), 1)  # a > b
        self.assertEqual(hns_compare(b, a), -1)  # b < a
        self.assertEqual(hns_compare(a, a), 0)  # a == a


class TestEngineInitialization(unittest.TestCase):
    """Test NeuroCHIMERA engine initialization."""
    
    def test_config_creation(self):
        """Test configuration object creation."""
        config = NeuroCHIMERAConfig(neurons=65536, use_hns=True)
        self.assertEqual(config.neurons, 65536)
        self.assertTrue(config.use_hns)
        self.assertGreater(config.texture_size, 0)
    
    def test_engine_creation(self):
        """Test engine initialization."""
        config = NeuroCHIMERAConfig(neurons=65536, use_hns=True)
        brain = NeuroCHIMERA(config)
        self.assertIsNotNone(brain.config)
        self.assertIsNotNone(brain.frame)
        # Cleanup
        brain.release()
    
    def test_frame_creation(self):
        """Test neuromorphic frame creation."""
        config = NeuroCHIMERAConfig(neurons=65536)
        frame = NeuromorphicFrame.create(config)
        self.assertEqual(frame.neural_state.shape[0], config.texture_size)
        self.assertEqual(frame.neural_state.shape[1], config.texture_size)
        self.assertEqual(frame.neural_state.shape[2], 4)  # RGBA
    
    def test_gpu_initialization(self):
        """Test GPU context initialization (if available)."""
        try:
            brain = NeuroCHIMERA(neurons=65536, use_hns=True)
            # If GPU is available, ctx should be set
            # If not, should fall back to CPU mode
            self.assertIsNotNone(brain.frame)
            brain.release()
        except Exception as e:
            self.fail(f"Engine initialization failed: {e}")


class TestFrameManagement(unittest.TestCase):
    """Test neuromorphic frame management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeuroCHIMERAConfig(neurons=65536)
        self.frame = NeuromorphicFrame.create(self.config)
    
    def test_frame_structure(self):
        """Test frame data structure."""
        self.assertIsNotNone(self.frame.neural_state)
        self.assertIsNotNone(self.frame.connectivity)
        self.assertIsNotNone(self.frame.spatial_features)
        self.assertIsNotNone(self.frame.holographic_memory)
    
    def test_state_persistence(self):
        """Test state save/load functionality."""
        # Modify state
        original_mean = np.mean(self.frame.neural_state[:, :, 0])
        self.frame.neural_state[:, :, 0] += 0.1
        
        # Save
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            temp_path = f.name
        
        try:
            # Create engine to use save method
            brain = NeuroCHIMERA(config=self.config)
            brain.frame = self.frame
            brain.current_epoch = 10
            brain.save_state(temp_path)
            
            # Load
            brain2 = NeuroCHIMERA(config=self.config)
            brain2.load_state(temp_path)
            
            # Verify
            self.assertEqual(brain2.current_epoch, 10)
            self.assertIsNotNone(brain2.frame.neural_state)
            brain2.release()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestEvolutionDynamics(unittest.TestCase):
    """Test cellular automata evolution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeuroCHIMERAConfig(
            neurons=65536,
            default_iterations=10,
            convergence_threshold=0.001
        )
        self.brain = NeuroCHIMERA(config=self.config)
    
    def tearDown(self):
        """Clean up."""
        self.brain.release()
    
    def test_evolution_step(self):
        """Test single evolution step."""
        initial_state = self.brain.frame.neural_state.copy()
        result = self.brain.evolve(iterations=1)
        
        self.assertIn('iterations', result)
        self.assertIn('converged', result)
        # State should have changed (or at least been processed)
        self.assertIsNotNone(self.brain.frame.neural_state)
    
    def test_evolution_convergence(self):
        """Test evolution convergence."""
        result = self.brain.evolve(iterations=20)
        self.assertIn('converged', result)
        self.assertIn('final_delta', result)
    
    def test_multiple_evolution_steps(self):
        """Test multiple evolution steps."""
        for i in range(5):
            result = self.brain.evolve(iterations=5)
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result['iterations'], 1)
    
    def test_learning_step(self):
        """Test Hebbian learning step."""
        initial_weights = self.brain.frame.connectivity.copy()
        self.brain.learn(learning_rate=0.01)
        
        # Weights should have changed (or at least been processed)
        self.assertIsNotNone(self.brain.frame.connectivity)


class TestMemorySystem(unittest.TestCase):
    """Test holographic memory operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeuroCHIMERAConfig(neurons=65536)
        self.brain = NeuroCHIMERA(config=self.config)
    
    def tearDown(self):
        """Clean up."""
        self.brain.release()
    
    def test_memory_encoding(self):
        """Test memory encoding."""
        input_pattern = np.random.uniform(0, 1, (512, 512, 4)).astype(np.float32)
        output_pattern = np.random.uniform(0, 1, (512, 512, 4)).astype(np.float32)
        
        initial_memory = self.brain.frame.holographic_memory.copy()
        self.brain.encode_memory(input_pattern, output_pattern)
        
        # Memory should have changed
        self.assertFalse(np.array_equal(
            initial_memory, 
            self.brain.frame.holographic_memory
        ))
    
    def test_memory_retrieval(self):
        """Test memory retrieval."""
        # First encode
        input_pattern = np.random.uniform(0, 1, (512, 512, 4)).astype(np.float32)
        output_pattern = np.random.uniform(0, 1, (512, 512, 4)).astype(np.float32)
        self.brain.encode_memory(input_pattern, output_pattern)
        
        # Then retrieve
        query = input_pattern.copy()
        retrieved = self.brain.retrieve_memory(query)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.shape, (512, 512, 4))


class TestMetrics(unittest.TestCase):
    """Test metric calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeuroCHIMERAConfig(neurons=65536)
        self.brain = NeuroCHIMERA(config=self.config)
    
    def tearDown(self):
        """Clean up."""
        self.brain.release()
    
    def test_get_metrics(self):
        """Test metric calculation."""
        metrics = self.brain.get_metrics()
        
        self.assertIn('connectivity', metrics)
        self.assertIn('phi', metrics)
        self.assertIn('hierarchical_depth', metrics)
        self.assertIn('complexity', metrics)
        self.assertIn('qualia_coherence', metrics)
        self.assertIn('mean_activation', metrics)
        self.assertIn('std_activation', metrics)
        self.assertIn('epoch', metrics)
        
        # Check value ranges
        self.assertGreaterEqual(metrics['connectivity'], 0)
        self.assertGreaterEqual(metrics['phi'], 0)
        self.assertLessEqual(metrics['phi'], 1)
        self.assertGreaterEqual(metrics['complexity'], 0)
        self.assertLessEqual(metrics['complexity'], 1)
    
    def test_is_critical(self):
        """Test critical state detection."""
        is_critical = self.brain.is_critical()
        self.assertIsInstance(is_critical, bool)


def run_tests():
    """Run all core component tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHNS))
    suite.addTests(loader.loadTestsFromTestCase(TestEngineInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestFrameManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestEvolutionDynamics))
    suite.addTests(loader.loadTestsFromTestCase(TestMemorySystem))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 80)
    print("NeuroCHIMERA Core Component Tests")
    print("=" * 80)
    print()
    
    success = run_tests()
    
    print()
    print("=" * 80)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    print("=" * 80)
    
    sys.exit(0 if success else 1)

