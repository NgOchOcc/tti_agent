#!/usr/bin/env python3
"""
TOS-RL Training Script with TTI Integration

Reads config from YAML files (scripts/config/main/*.yaml)
Integrates TOS-RL training with TTI trajectory collection
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
import numpy as np

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

try:
    from tti.tos_rl import (
        TOSRLTrainer, TOSRLConfig,
        BudgetState, TokenModeFormatter
    )
except ImportError as e:
    print(f"Error importing TOS-RL modules: {e}")
    print("Make sure you're in the correct directory and all files exist")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TOSRLTrainingPipeline:
    """Training pipeline for TOS-RL with TTI integration."""

    def __init__(self, config_dict: Dict[str, Any], output_dir: Path):
        """
        Initialize training pipeline.

        Args:
            config_dict: Config from YAML file
            output_dir: Directory for logs and checkpoints
        """
        self.config_dict = config_dict
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract TOS-RL parameters from config
        self.epochs = config_dict.get('epochs', 10)
        self.batch_size = config_dict.get('batch_size', 4)
        self.grad_accum_steps = config_dict.get('grad_accum_steps', 2)
        self.eval_freq = config_dict.get('eval_freq', 1)
        self.save_freq = config_dict.get('save_freq', 1)

        # Create TOS-RL config
        self.tosrl_config = TOSRLConfig(
            group_size=config_dict.get('group_size', 4),
            learning_rate=config_dict.get('lm_lr', 1e-5),
            mode_token_weight=config_dict.get('mode_weight', 2.0),
            use_branching=config_dict.get('use_branching', False),
        )

        # Create trainer
        self.trainer = TOSRLTrainer(self.tosrl_config)

        # Checkpoints directory
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized training pipeline")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"TOS-RL config: {self.tosrl_config}")

    def generate_mock_trajectory(self, task_id: str, task_idx: int) -> Dict[str, Any]:
        """Generate mock trajectory (replace with actual TTI agent)."""
        return {
            "task_id": task_id,
            "success": np.random.randint(0, 2),
            "num_steps": np.random.randint(3, 20),
            "num_tokens": np.random.randint(100, 500),
            "num_loops": np.random.randint(0, 3),
            "num_bad_actions": np.random.randint(0, 2),
            "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{self.epochs}")
        logger.info(f"{'='*60}")

        epoch_metrics = {
            "total_loss": [],
            "policy_loss": [],
            "mode_loss": [],
            "success_rate": [],
        }

        # Sample tasks
        num_tasks = 16  # Mock value
        for task_idx in range(num_tasks):
            task_id = f"task_{epoch}_{task_idx}"

            # Collect K trajectories
            trajectories = []
            for k in range(self.tosrl_config.group_size):
                traj = self.generate_mock_trajectory(task_id, task_idx)
                trajectories.append(traj)

            # Process trajectories (compute utilities and advantages)
            try:
                utilities, costs = self.trainer.process_trajectories(trajectories)
                advantages = self.trainer.compute_group_advantages(utilities)

                # Prepare batch
                batch = {
                    "log_probs": torch.randn(len(trajectories)),
                    "log_probs_old": torch.randn(len(trajectories)),
                    "log_probs_ref": torch.randn(len(trajectories)),
                    "advantages": torch.tensor(advantages, dtype=torch.float32),
                    "mode_log_probs": torch.randn(len(trajectories)),
                    "mode_probs": torch.ones(len(trajectories), 3) / 3,
                }

                # Training step
                metrics = self.trainer.training_step(batch)

                # Track metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])

                success_rate = sum(t['success'] for t in trajectories) / len(trajectories)
                epoch_metrics['success_rate'].append(success_rate)

                if (task_idx + 1) % 4 == 0:
                    logger.info(
                        f"  Task {task_idx + 1}/{num_tasks}: "
                        f"loss={metrics.get('total_loss', 0):.4f}, "
                        f"success={success_rate:.2%}"
                    )

            except Exception as e:
                logger.error(f"Error processing task {task_id}: {e}")
                continue

        # Compute epoch statistics
        epoch_stats = {}
        for key, values in epoch_metrics.items():
            if values:
                avg = np.mean(values)
                epoch_stats[key] = avg
                logger.info(f"  {key}: {avg:.4f}")

        return epoch_stats

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "metrics": metrics,
            "config": asdict(self.tosrl_config),
        }

        checkpoint_path = self.checkpoints_dir / f"epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if not hasattr(self, "best_success") or metrics.get("success_rate", 0) > self.best_success:
            self.best_success = metrics.get("success_rate", 0)
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Updated best checkpoint: {best_path}")

    def train(self):
        """Run training loop."""
        logger.info(f"Starting training for {self.epochs} epochs...")
        self.best_success = 0

        all_results = []

        for epoch in range(self.epochs):
            # Train
            epoch_metrics = self.train_epoch(epoch)

            # Save
            if (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch, epoch_metrics)

            all_results.append({
                "epoch": epoch + 1,
                "metrics": epoch_metrics,
            })

        # Save results
        results_file = self.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved results: {results_file}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed!")
        logger.info(f"Checkpoints: {self.checkpoints_dir}")
        logger.info(f"Results: {results_file}")
        logger.info(f"{'='*60}")


def load_config(config_file: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="TOS-RL Training with TTI")

    # Config
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # Experiment
    parser.add_argument("--experiment", type=str, required=True, help="Experiment ID")
    parser.add_argument("--dataset", type=str, default="webarena", help="Dataset name")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    # Training parameters (override config)
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--grad-accum-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--eval-freq", type=int, help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, help="Save frequency")

    # TOS-RL parameters
    parser.add_argument("--mode", type=str, default="tosrl_train", help="Training mode")
    parser.add_argument("--group-size", type=int, help="Group size")
    parser.add_argument("--mode-weight", type=float, help="Mode token weight")
    parser.add_argument("--branching", action="store_true", help="Enable branching")

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Override with command-line arguments
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.grad_accum_steps:
        config["grad_accum_steps"] = args.grad_accum_steps
    if args.eval_freq:
        config["eval_freq"] = args.eval_freq
    if args.save_freq:
        config["save_freq"] = args.save_freq
    if args.group_size:
        config["group_size"] = args.group_size
    if args.mode_weight:
        config["mode_weight"] = args.mode_weight
    if args.branching:
        config["use_branching"] = True

    # Create pipeline
    pipeline = TOSRLTrainingPipeline(config, args.output_dir)

    # Train
    pipeline.train()


if __name__ == "__main__":
    main()
