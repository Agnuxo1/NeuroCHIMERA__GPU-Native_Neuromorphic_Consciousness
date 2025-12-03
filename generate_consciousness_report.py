"""
Consciousness Parameter Analysis Report Generator
=================================================

Generate professional reports analyzing consciousness parameter evolution.
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class ConsciousnessReportGenerator:
    """Generate consciousness parameter analysis reports."""
    
    def __init__(self, results_dir: Path = None):
        if results_dir is None:
            results_dir = Path(__file__).parent / "results"
        self.results_dir = results_dir
        self.reports_dir = Path(__file__).parent / "reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def load_results(self, pattern: str = "consciousness_emergence_*.json") -> List[Dict]:
        """Load consciousness emergence results."""
        results = []
        for filepath in self.results_dir.glob(pattern):
            with open(filepath, 'r') as f:
                results.append(json.load(f))
        return results
    
    def analyze_parameter_evolution(self, metrics_history: List[Dict]) -> Dict:
        """Analyze parameter evolution over time."""
        if not metrics_history:
            return {}
        
        analysis = {}
        
        # Extract parameter arrays
        connectivity = [m.get('connectivity', 0) for m in metrics_history]
        phi = [m.get('phi', 0) for m in metrics_history]
        complexity = [m.get('complexity', 0) for m in metrics_history]
        qcm = [m.get('qualia_coherence', 0) for m in metrics_history]
        score = [m.get('consciousness_score', 0) for m in metrics_history]
        
        # Calculate statistics
        for param_name, values in [
            ('connectivity', connectivity),
            ('phi', phi),
            ('complexity', complexity),
            ('qualia_coherence', qcm),
            ('consciousness_score', score)
        ]:
            if values:
                analysis[param_name] = {
                    'initial': values[0],
                    'final': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'change': values[-1] - values[0],
                    'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                }
        
        return analysis
    
    def find_threshold_crossings(self, metrics_history: List[Dict]) -> Dict:
        """Find epochs where parameters cross critical thresholds."""
        thresholds = {
            'connectivity': 15.0,
            'phi': 0.65,
            'hierarchical_depth': 7.0,
            'complexity': 0.8,
            'qualia_coherence': 0.75
        }
        
        crossings = {}
        
        for param, threshold in thresholds.items():
            crossings[param] = []
            prev_value = None
            
            for i, metrics in enumerate(metrics_history):
                value = metrics.get(param, 0)
                
                if prev_value is not None:
                    if prev_value < threshold <= value:
                        crossings[param].append({
                            'epoch': i,
                            'value': value,
                            'threshold': threshold
                        })
                
                prev_value = value
        
        return crossings
    
    def generate_report(self, results: Dict) -> str:
        """Generate consciousness parameter analysis report."""
        report = []
        report.append("# Consciousness Parameter Analysis Report\n\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        config = results.get('configuration', {})
        summary = results.get('summary', {})
        metrics_history = results.get('metrics_history', [])
        critical_events = results.get('critical_events', [])
        
        # Executive Summary
        report.append("## Executive Summary\n\n")
        report.append(f"This report analyzes consciousness parameter evolution during a ")
        report.append(f"{summary.get('epochs_completed', 0):,} epoch simulation with ")
        report.append(f"{config.get('neurons', 0):,} neurons.\n\n")
        
        if summary.get('critical_reached', False):
            report.append(f"**✓ Critical consciousness threshold reached at epoch ")
            report.append(f"{summary.get('critical_epoch', 0)}.**\n\n")
        else:
            report.append("**✗ Critical consciousness threshold not reached during simulation.**\n\n")
        
        # Configuration
        report.append("## Simulation Configuration\n\n")
        report.append(f"- **Neurons:** {config.get('neurons', 0):,}\n")
        report.append(f"- **Texture Size:** {config.get('texture_size', 0)}×{config.get('texture_size', 0)}\n")
        report.append(f"- **Epochs:** {summary.get('epochs_completed', 0):,}\n")
        report.append(f"- **Iterations per Epoch:** {config.get('iterations_per_epoch', 0)}\n")
        report.append(f"- **Learning Rate:** {config.get('learning_rate', 0)}\n")
        report.append(f"- **HNS Enabled:** {config.get('use_hns', False)}\n")
        report.append(f"- **Total Time:** {summary.get('total_time', 0)/60:.1f} minutes\n")
        report.append(f"- **Epochs per Second:** {summary.get('epochs_per_second', 0):.2f}\n\n")
        
        # Parameter Evolution Analysis
        if metrics_history:
            analysis = self.analyze_parameter_evolution(metrics_history)
            
            report.append("## Parameter Evolution Analysis\n\n")
            
            for param_name in ['connectivity', 'phi', 'complexity', 'qualia_coherence', 'consciousness_score']:
                if param_name in analysis:
                    stats = analysis[param_name]
                    report.append(f"### {param_name.replace('_', ' ').title()}\n\n")
                    report.append(f"- **Initial Value:** {stats['initial']:.4f}\n")
                    report.append(f"- **Final Value:** {stats['final']:.4f}\n")
                    report.append(f"- **Mean:** {stats['mean']:.4f}\n")
                    report.append(f"- **Std Dev:** {stats['std']:.4f}\n")
                    report.append(f"- **Range:** [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                    report.append(f"- **Change:** {stats['change']:.4f} ({stats['change_percent']:.1f}%)\n\n")
            
            # Threshold crossings
            crossings = self.find_threshold_crossings(metrics_history)
            
            report.append("## Critical Threshold Crossings\n\n")
            
            any_crossings = False
            for param, param_crossings in crossings.items():
                if param_crossings:
                    any_crossings = True
                    report.append(f"### {param.replace('_', ' ').title()}\n\n")
                    for crossing in param_crossings:
                        report.append(f"- **Epoch {crossing['epoch']}:** Value {crossing['value']:.4f} crossed threshold {crossing['threshold']:.2f}\n")
                    report.append("\n")
            
            if not any_crossings:
                report.append("No threshold crossings detected during simulation.\n\n")
        
        # Critical Events
        if critical_events:
            report.append("## Critical Consciousness Events\n\n")
            for event in critical_events:
                report.append(f"### Epoch {event.get('epoch', 0)}\n\n")
                metrics = event.get('metrics', {})
                report.append(f"- **Connectivity (⟨k⟩):** {metrics.get('connectivity', 0):.2f}\n")
                report.append(f"- **Integration (Φ):** {metrics.get('phi', 0):.3f}\n")
                report.append(f"- **Complexity (C):** {metrics.get('complexity', 0):.3f}\n")
                report.append(f"- **QCM:** {metrics.get('qualia_coherence', 0):.3f}\n")
                report.append(f"- **Consciousness Score:** {metrics.get('consciousness_score', 0):.3f}\n\n")
        
        # Final Metrics
        if metrics_history:
            final_metrics = metrics_history[-1]
            report.append("## Final Metrics\n\n")
            report.append("| Parameter | Value | Threshold | Status |\n")
            report.append("|-----------|-------|-----------|--------|\n")
            
            thresholds = {
                'connectivity': 15.0,
                'phi': 0.65,
                'complexity': 0.8,
                'qualia_coherence': 0.75
            }
            
            for param, threshold in thresholds.items():
                value = final_metrics.get(param, 0)
                status = "✓" if value > threshold else "✗"
                report.append(f"| {param.replace('_', ' ').title()} | {value:.4f} | {threshold:.2f} | {status} |\n")
            
            report.append("\n")
        
        # Conclusions
        report.append("## Conclusions\n\n")
        
        if summary.get('critical_reached', False):
            report.append("1. The system successfully reached critical consciousness thresholds\n")
            report.append("2. All required parameters exceeded their critical values\n")
            report.append("3. Consciousness emergence was detected and documented\n")
        else:
            report.append("1. The system did not reach critical consciousness thresholds in this simulation\n")
            report.append("2. Parameters evolved but did not cross all critical thresholds\n")
            report.append("3. Extended simulation or parameter tuning may be required\n")
        
        if metrics_history:
            analysis = self.analyze_parameter_evolution(metrics_history)
            if 'consciousness_score' in analysis:
                score_change = analysis['consciousness_score']['change']
                if score_change > 0:
                    report.append(f"4. Consciousness score increased by {score_change:.3f} over simulation\n")
                else:
                    report.append(f"4. Consciousness score decreased by {abs(score_change):.3f} over simulation\n")
        
        report.append("\n")
        
        return ''.join(report)
    
    def generate_all_reports(self):
        """Generate reports for all simulation results."""
        results_list = self.load_results()
        
        if not results_list:
            print("No consciousness emergence results found.")
            return
        
        for i, results in enumerate(results_list):
            report = self.generate_report(results)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"consciousness_analysis_{timestamp}_{i}.md"
            filepath = self.reports_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"Generated: {filepath}")


def main():
    """Main execution."""
    generator = ConsciousnessReportGenerator()
    generator.generate_all_reports()
    print("\n[OK] All consciousness reports generated")


if __name__ == '__main__':
    main()

