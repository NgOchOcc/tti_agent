#!/usr/bin/env python3
"""
Pareto Analysis for TOS-RL Evaluation Results

Analyzes success-cost frontier across different cost preferences
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ParetoAnalyzer:
    """Analyzes Pareto frontier from evaluation results."""

    def __init__(self, results_dir: Path):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)

    def load_metrics(self, cost_preference: str) -> Dict[str, Any]:
        """Load metrics for a cost preference."""
        metrics_file = self.results_dir / f"metrics_{cost_preference}.json"

        if not metrics_file.exists():
            logger.warning(f"Metrics file not found: {metrics_file}")
            return None

        with open(metrics_file, "r") as f:
            return json.load(f)

    def analyze(self) -> Dict[str, Any]:
        """Analyze Pareto frontier."""
        logger.info("Analyzing Pareto frontier...")

        # Load metrics for each cost preference
        cost_prefs = ["high_efficiency", "balanced", "high_success"]
        results = {}

        for pref in cost_prefs:
            metrics = self.load_metrics(pref)
            if metrics:
                results[pref] = metrics
                logger.info(f"\n{pref}:")
                logger.info(f"  Success: {metrics['success_rate']:.2%}")
                logger.info(f"  Avg Steps: {metrics['mean_steps']:.1f}")
                logger.info(f"  Avg Tokens: {metrics['mean_tokens']:.0f}")

        # Compute Pareto metrics
        if results:
            pareto = self._compute_pareto_metrics(results)
            return pareto
        else:
            logger.warning("No metrics found for Pareto analysis")
            return None

    def _compute_pareto_metrics(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute Pareto frontier metrics."""
        pareto = {
            "by_cost_preference": {},
            "frontier": [],
        }

        # Store each preference's results
        for pref, metrics in results.items():
            pareto["by_cost_preference"][pref] = {
                "success_rate": metrics["success_rate"],
                "mean_steps": metrics["mean_steps"],
                "mean_tokens": metrics["mean_tokens"],
                "mode_distribution": metrics.get("mode_distribution", {}),
            }

            # Add to frontier
            pareto["frontier"].append({
                "cost_preference": pref,
                "success": metrics["success_rate"],
                "cost_steps": metrics["mean_steps"],
                "cost_tokens": metrics["mean_tokens"],
            })

        # Compute efficiency metrics
        if len(results) >= 2:
            # Success improvement from high_efficiency to high_success
            if "high_efficiency" in results and "high_success" in results:
                success_diff = (
                    results["high_success"]["success_rate"] -
                    results["high_efficiency"]["success_rate"]
                )
                steps_diff = (
                    results["high_efficiency"]["mean_steps"] -
                    results["high_success"]["mean_steps"]
                )

                pareto["efficiency_metrics"] = {
                    "success_improvement": success_diff,
                    "step_difference": steps_diff,
                    "efficiency_ratio": success_diff / (steps_diff + 1e-6),
                }

        return pareto

    def save_analysis(self, pareto: Dict[str, Any], output_file: Path):
        """Save Pareto analysis results."""
        with open(output_file, "w") as f:
            json.dump(pareto, f, indent=2)
        logger.info(f"Saved Pareto analysis: {output_file}")

    def print_summary(self, pareto: Dict[str, Any]):
        """Print summary of Pareto analysis."""
        logger.info("\n" + "="*60)
        logger.info("PARETO FRONTIER SUMMARY")
        logger.info("="*60)

        for item in pareto["frontier"]:
            logger.info(f"\n{item['cost_preference']}:")
            logger.info(f"  Success Rate: {item['success']:.2%}")
            logger.info(f"  Avg Steps: {item['cost_steps']:.1f}")
            logger.info(f"  Avg Tokens: {item['cost_tokens']:.0f}")

        if "efficiency_metrics" in pareto:
            logger.info(f"\nEfficiency Metrics:")
            for key, val in pareto["efficiency_metrics"].items():
                logger.info(f"  {key}: {val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Pareto Analysis for TOS-RL Evaluation")

    parser.add_argument("--results-dir", type=str, required=True,
                       help="Directory containing evaluation results")
    parser.add_argument("--experiment-id", type=str, help="Experiment ID")
    parser.add_argument("--output", type=str, help="Output file for analysis")

    args = parser.parse_args()

    # Create analyzer
    analyzer = ParetoAnalyzer(args.results_dir)

    # Analyze
    pareto = analyzer.analyze()

    if pareto:
        # Save
        output_file = args.output or Path(args.results_dir) / "pareto_analysis.json"
        analyzer.save_analysis(pareto, Path(output_file))

        # Print summary
        analyzer.print_summary(pareto)

        logger.info("\n" + "="*60)
        logger.info("Pareto analysis complete!")
        logger.info("="*60)
    else:
        logger.error("No metrics found for analysis")
        sys.exit(1)


if __name__ == "__main__":
    main()
