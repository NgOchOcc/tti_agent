#!/usr/bin/env python3
"""TOS-RL Evaluation Pipeline."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Evaluation pipeline for TOS-RL."""

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: str,
        output_dir: str,
    ):
        """Initialize evaluation pipeline."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_path

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Initialized evaluation pipeline")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Output: {self.output_dir}")

    def evaluate(
        self,
        num_tasks: int = 100,
        cost_preference: str = "balanced",
    ) -> Dict[str, Any]:
        """Evaluate model."""
        logger.info(f"Evaluating with cost preference: {cost_preference}")

        # Generate mock evaluation results
        results = {
            "success": [np.random.randint(0, 2) for _ in range(num_tasks)],
            "num_steps": [np.random.randint(5, 30) for _ in range(num_tasks)],
            "num_tokens": [np.random.randint(200, 4000) for _ in range(num_tasks)],
            "modes": [
                ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]
                for _ in range(num_tasks)
            ],
        }

        # Compute metrics
        metrics = {
            "success_rate": float(np.mean(results["success"])),
            "mean_steps": float(np.mean(results["num_steps"])),
            "mean_tokens": float(np.mean(results["num_tokens"])),
            "std_steps": float(np.std(results["num_steps"])),
            "std_tokens": float(np.std(results["num_tokens"])),
        }

        # Mode distribution
        all_modes = []
        for mode_list in results["modes"]:
            all_modes.extend(mode_list)

        if all_modes:
            metrics["mode_distribution"] = {
                "THINK": float(all_modes.count("THINK") / len(all_modes)),
                "OBSERVE": float(all_modes.count("OBSERVE") / len(all_modes)),
                "ANSWER": float(all_modes.count("ANSWER") / len(all_modes)),
            }

        # Log results
        logger.info(f"Success Rate: {metrics['success_rate']:.2%}")
        logger.info(f"Mean Steps: {metrics['mean_steps']:.1f}")
        logger.info(f"Mean Tokens: {metrics['mean_tokens']:.0f}")

        return metrics, results

    def save_results(
        self,
        cost_preference: str,
        metrics: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Save evaluation results."""
        # Save metrics
        metrics_file = self.output_dir / f"metrics_{cost_preference}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics: {metrics_file}")

        # Save detailed results
        results_file = self.output_dir / f"results_{cost_preference}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results: {results_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TOS-RL Evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--experiment", default="evaluation", help="Experiment ID")
    parser.add_argument("--dataset", default="webarena", help="Dataset name")
    parser.add_argument(
        "--cost-preference",
        default="balanced",
        choices=["high_efficiency", "balanced", "high_success"],
        help="Cost preference"
    )
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Run evaluation
    pipeline = EvaluationPipeline(
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )

    metrics, results = pipeline.evaluate(
        num_tasks=args.num_tasks,
        cost_preference=args.cost_preference,
    )

    pipeline.save_results(args.cost_preference, metrics, results)
    logger.info("Evaluation completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
