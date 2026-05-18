#!/usr/bin/env python3
"""
TOS-RL Evaluation Script with TTI Integration

Evaluates trained model on test set with different cost preferences
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import numpy as np

from tti.tos_rl import (
    TOSRLInference,
    BudgetState,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TOSRLEvaluationPipeline:
    """Evaluation pipeline for TOS-RL with TTI integration."""

    def __init__(self, config_dict: Dict[str, Any], output_dir: Path):
        """
        Initialize evaluation pipeline.

        Args:
            config_dict: Config from YAML file
            output_dir: Directory for logs and results
        """
        self.config_dict = config_dict
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create inference controller
        self.inference = TOSRLInference(
            max_steps=config_dict.get('max_iter', 30),
            max_tokens=config_dict.get('max_new_tokens', 4096),
        )

        logger.info(f"Initialized evaluation pipeline")
        logger.info(f"Output directory: {self.output_dir}")

    def generate_mock_task(self, task_idx: int) -> Dict[str, Any]:
        """Generate mock task (replace with actual WebArena task)."""
        return {
            "id": f"task_{task_idx}",
            "description": f"Find information about item {task_idx}",
        }

    def evaluate_task(self, task: Dict[str, Any], budget: BudgetState) -> Dict[str, Any]:
        """Evaluate single task."""
        # Mock inference (replace with actual LLM inference)
        result = {
            "task_id": task["id"],
            "success": np.random.randint(0, 2),
            "num_steps": np.random.randint(5, 30),
            "num_tokens": np.random.randint(200, 4000),
            "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],
            "final_answer": "mock answer",
        }
        return result

    def evaluate(self, num_tasks: int, cost_preference: str) -> Dict[str, Any]:
        """Evaluate on test set."""
        logger.info(f"\nEvaluating with cost preference: {cost_preference}")
        logger.info(f"Number of tasks: {num_tasks}")

        # Set budget based on cost preference
        if cost_preference == "high_efficiency":
            budget = BudgetState(env_interactions=10, tokens=2000)
        elif cost_preference == "high_success":
            budget = BudgetState(env_interactions=50, tokens=8192)
        else:  # balanced
            budget = BudgetState(env_interactions=30, tokens=4096)

        # Evaluate
        results = {
            "success": [],
            "num_steps": [],
            "num_tokens": [],
            "modes": [],
        }

        for task_idx in range(num_tasks):
            task = self.generate_mock_task(task_idx)

            try:
                result = self.evaluate_task(task, budget)

                results["success"].append(result["success"])
                results["num_steps"].append(result["num_steps"])
                results["num_tokens"].append(result["num_tokens"])
                results["modes"].append(result["modes"])

                if (task_idx + 1) % 10 == 0:
                    logger.info(f"  Task {task_idx + 1}/{num_tasks}")

            except Exception as e:
                logger.error(f"Error evaluating task {task_idx}: {e}")
                continue

        # Compute metrics
        metrics = {
            "success_rate": np.mean(results["success"]),
            "mean_steps": np.mean(results["num_steps"]),
            "mean_tokens": np.mean(results["num_tokens"]),
            "std_steps": np.std(results["num_steps"]),
            "std_tokens": np.std(results["num_tokens"]),
        }

        # Analyze modes
        all_modes = []
        for mode_list in results["modes"]:
            all_modes.extend(mode_list)

        mode_dist = {
            "THINK": all_modes.count("THINK") / len(all_modes) if all_modes else 0,
            "OBSERVE": all_modes.count("OBSERVE") / len(all_modes) if all_modes else 0,
            "ANSWER": all_modes.count("ANSWER") / len(all_modes) if all_modes else 0,
        }
        metrics["mode_distribution"] = mode_dist

        # Log results
        logger.info(f"\nResults for {cost_preference}:")
        logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
        logger.info(f"  Mean Steps: {metrics['mean_steps']:.1f}")
        logger.info(f"  Mean Tokens: {metrics['mean_tokens']:.0f}")
        logger.info(f"  Mode Distribution:")
        logger.info(f"    THINK: {mode_dist['THINK']:.1%}")
        logger.info(f"    OBSERVE: {mode_dist['OBSERVE']:.1%}")
        logger.info(f"    ANSWER: {mode_dist['ANSWER']:.1%}")

        return metrics, results

    def save_results(self, cost_preference: str, metrics: Dict[str, Any], results: Dict[str, Any]):
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


def load_config(config_file: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="TOS-RL Evaluation with TTI")

    # Config and checkpoint
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Experiment
    parser.add_argument("--experiment", type=str, required=True, help="Experiment ID")
    parser.add_argument("--dataset", type=str, default="webarena", help="Dataset name")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    # Evaluation parameters
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks to evaluate")
    parser.add_argument("--cost-preference", type=str, default="balanced",
                       choices=["high_efficiency", "balanced", "high_success"],
                       help="Cost preference")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--mode", type=str, default="tosrl_eval", help="Evaluation mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    logger.info(f"Using checkpoint: {args.checkpoint}")

    # Create evaluation pipeline
    pipeline = TOSRLEvaluationPipeline(config, args.output_dir)

    # Evaluate
    metrics, results = pipeline.evaluate(args.num_tasks, args.cost_preference)

    # Save results
    pipeline.save_results(args.cost_preference, metrics, results)

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
