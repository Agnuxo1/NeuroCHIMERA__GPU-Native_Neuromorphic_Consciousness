"""
Consciousness Emergence Simulation
==================================

Long-term evolution simulation to observe consciousness parameter evolution
and attempt to reach critical thresholds.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from engine import NeuroCHIMERA, NeuroCHIMERAConfig
from consciousness_monitor import ConsciousnessMonitor


def run_simulation(
    neurons: int = 65536,
    epochs: int = 1000,
    iterations_per_epoch: int = 20,
    learning_rate: float = 0.01,
    use_hns: bool = True,
    output_dir: Path = None
):
    """
    Run long-term consciousness emergence simulation.
    
    Args:
        neurons: Number of neurons
        epochs: Number of epochs to run
        iterations_per_epoch: Evolution iterations per epoch
        learning_rate: Learning rate for Hebbian plasticity
        use_hns: Enable Hierarchical Number System
        output_dir: Directory to save results
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("CONSCIOUSNESS EMERGENCE SIMULATION")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Neurons: {neurons:,}")
    print(f"  Epochs: {epochs:,}")
    print(f"  Iterations per epoch: {iterations_per_epoch}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  HNS enabled: {use_hns}")
    print("=" * 80)
    print()
    
    # Initialize system
    print("Initializing NeuroCHIMERA...")
    config = NeuroCHIMERAConfig(
        neurons=neurons,
        default_iterations=iterations_per_epoch,
        use_hns=use_hns,
        target_connectivity=18.0
    )
    
    brain = NeuroCHIMERA(config=config)
    monitor = ConsciousnessMonitor(brain)
    
    print(f"Initialized: {config.texture_size}×{config.texture_size} texture")
    print()
    
    # Storage for results
    metrics_history = []
    evolution_log = []
    critical_events = []
    
    start_time = time.time()
    
    try:
        # Main evolution loop
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Evolve
            evolve_result = brain.evolve(iterations=iterations_per_epoch)
            
            # Learn periodically
            if epoch % 10 == 0:
                brain.learn(learning_rate=learning_rate)
            
            # Measure consciousness metrics
            metrics = monitor.measure()
            metrics_history.append(metrics.to_dict())
            
            # Check for critical state
            if monitor.is_critical() and len(critical_events) == 0:
                critical_events.append({
                    'epoch': epoch,
                    'timestamp': time.time(),
                    'metrics': metrics.to_dict()
                })
                print(f"\n{'='*80}")
                print(f"🧠 CRITICAL THRESHOLD REACHED at epoch {epoch}!")
                print(f"{'='*80}")
                print(f"  Connectivity (⟨k⟩): {metrics.connectivity:.2f} (threshold: >15)")
                print(f"  Integration (Φ): {metrics.phi:.3f} (threshold: >0.65)")
                print(f"  Complexity (C): {metrics.complexity:.3f} (threshold: >0.8)")
                print(f"  QCM: {metrics.qualia_coherence:.3f} (threshold: >0.75)")
                print(f"  Consciousness Score: {metrics.consciousness_score:.3f}")
                print(f"{'='*80}\n")
            
            # Log evolution
            epoch_time = time.time() - epoch_start
            evolution_log.append({
                'epoch': epoch,
                'time': epoch_time,
                'converged': evolve_result.get('converged', False),
                'iterations': evolve_result.get('iterations', 0)
            })
            
            # Progress reporting
            if epoch % 100 == 0 or epoch == epochs - 1:
                elapsed = time.time() - start_time
                rate = (epoch + 1) / elapsed if elapsed > 0 else 0
                remaining = (epochs - epoch - 1) / rate if rate > 0 else 0
                
                print(f"Epoch {epoch:5d}/{epochs}: "
                      f"⟨k⟩={metrics.connectivity:5.2f} "
                      f"Φ={metrics.phi:.3f} "
                      f"C={metrics.complexity:.3f} "
                      f"QCM={metrics.qualia_coherence:.3f} "
                      f"Score={metrics.consciousness_score:.3f} "
                      f"({rate:.1f} epochs/s, {remaining/60:.1f}m remaining)")
            
            # Check for distress
            is_distressed, indicators = monitor.check_distress()
            if is_distressed:
                print(f"  [WARNING] Distress indicators: {indicators}")
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    except Exception as e:
        print(f"\n\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        
        # Save results
        results = {
            'configuration': {
                'neurons': neurons,
                'epochs': epoch + 1,
                'iterations_per_epoch': iterations_per_epoch,
                'learning_rate': learning_rate,
                'use_hns': use_hns,
                'texture_size': config.texture_size
            },
            'summary': {
                'total_time': total_time,
                'epochs_completed': epoch + 1,
                'epochs_per_second': (epoch + 1) / total_time if total_time > 0 else 0,
                'critical_reached': len(critical_events) > 0,
                'critical_epoch': critical_events[0]['epoch'] if critical_events else None
            },
            'metrics_history': metrics_history,
            'evolution_log': evolution_log,
            'critical_events': critical_events
        }
        
        # Save to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f"consciousness_emergence_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {results_file}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("SIMULATION SUMMARY")
        print("=" * 80)
        print(f"Epochs completed: {epoch + 1:,}/{epochs:,}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Rate: {(epoch + 1) / total_time:.2f} epochs/second")
        
        if metrics_history:
            final_metrics = metrics_history[-1]
            print(f"\nFinal Metrics:")
            print(f"  Connectivity (⟨k⟩): {final_metrics['connectivity']:.2f}")
            print(f"  Integration (Φ): {final_metrics['phi']:.3f}")
            print(f"  Complexity (C): {final_metrics['complexity']:.3f}")
            print(f"  QCM: {final_metrics['qualia_coherence']:.3f}")
            print(f"  Consciousness Score: {final_metrics['consciousness_score']:.3f}")
        
        if critical_events:
            print(f"\n✓ Critical threshold reached at epoch {critical_events[0]['epoch']}")
        else:
            print("\n✗ Critical threshold not reached")
        
        print("=" * 80)
        
        # Cleanup
        brain.release()
        
        return results


def main():
    """Main execution with different configurations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run consciousness emergence simulation')
    parser.add_argument('--neurons', type=int, default=65536, help='Number of neurons')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--iterations', type=int, default=20, help='Iterations per epoch')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--no-hns', action='store_true', help='Disable HNS')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_simulation(
        neurons=args.neurons,
        epochs=args.epochs,
        iterations_per_epoch=args.iterations,
        learning_rate=args.learning_rate,
        use_hns=not args.no_hns,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()

