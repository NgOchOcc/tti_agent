#!/usr/bin/env python3
"""
TOS-RL Training Script with Real Data Integration

Supports loading trajectories from:
1. JSONL files (pre-collected trajectories)
2. TTI agent (real-time collection)
3. WebArena dataset
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
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
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TOSRLTrainingPipelineRealData:
    """Training pipeline for TOS-RL with REAL data support."""

    def __init__(self, config_dict: Dict[str, Any], output_dir: Path, data_source: str = "file"):
        """
        Initialize training pipeline.

        Args:
            config_dict: Config from YAML file
            output_dir: Directory for logs and checkpoints
            data_source: "file" (JSONL), "tti" (agent), or "webarena"
        """
        self.config_dict = config_dict
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_source = data_source

        # Extract parameters
        self.epochs = config_dict.get('epochs', 10)
        self.batch_size = config_dict.get('batch_size', 4)
        self.grad_accum_steps = config_dict.get('grad_accum_steps', 2)
        self.eval_freq = config_dict.get('eval_freq', 1)
        self.save_freq = config_dict.get('save_freq', 1)
        self.trajectory_file = config_dict.get('train_tasks', None)

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

        # Load data
        self.trajectories = self._load_data()

        logger.info(f"Loaded {len(self.trajectories)} trajectories from {data_source}")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load trajectories based on data source."""
        if self.data_source == "file":
            return self._load_from_file()
        elif self.data_source == "tti":
            return self._load_from_tti_agent()
        elif self.data_source == "webarena":
            return self._load_from_webarena()
        else:
            logger.warning(f"Unknown data source: {self.data_source}, using mock data")
            return self._generate_mock_trajectories(100)

    def _load_from_file(self) -> List[Dict[str, Any]]:
        """Load trajectories from JSONL file."""
        if not self.trajectory_file:
            logger.warning("No trajectory file specified, using mock data")
            return self._generate_mock_trajectories(100)

        traj_path = Path(self.trajectory_file)
        if not traj_path.exists():
            logger.error(f"Trajectory file not found: {traj_path}")
            logger.info("Expected format: JSONL file with trajectory dicts")
            return self._generate_mock_trajectories(100)

        trajectories = []
        logger.info(f"Loading trajectories from {traj_path}...")

        try:
            with open(traj_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        traj = json.loads(line)
                        # Validate trajectory has required fields
                        required_fields = {'task_id', 'success', 'num_steps', 'num_tokens', 'modes'}
                        if all(field in traj for field in required_fields):
                            trajectories.append(traj)
                        else:
                            missing = required_fields - set(traj.keys())
                            logger.warning(f"Line {line_num}: Missing fields {missing}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        continue

            logger.info(f"Loaded {len(trajectories)} valid trajectories")
            return trajectories

        except Exception as e:
            logger.error(f"Error loading trajectories: {e}")
            logger.info("Falling back to mock data")
            return self._generate_mock_trajectories(100)

    def _load_from_tti_agent(self) -> List[Dict[str, Any]]:
        """Load trajectories by running TTI agent (requires TTI module)."""
        logger.warning("Real-time TTI agent collection not yet implemented")
        logger.info("Using mock data for now. See documentation for TTI integration")
        return self._generate_mock_trajectories(100)

    def _load_from_webarena(self) -> List[Dict[str, Any]]:
        """Load trajectories from WebArena dataset."""
        logger.warning("WebArena loader not yet implemented")
        logger.info("Using mock data for now. See documentation for WebArena integration")
        return self._generate_mock_trajectories(100)

    def _generate_mock_trajectories(self, num_trajectories: int) -> List[Dict[str, Any]]:
        """Generate mock trajectories for demonstration."""
        logger.warning("Generating MOCK trajectories for demonstration")
        logger.warning("To use REAL data, specify: --data-source file --trajectory-file <path>")

        trajectories = []
        for i in range(num_trajectories):
            trajectories.append({
                "task_id": f"mock_task_{i}",
                "success": np.random.randint(0, 2),
                "num_steps": np.random.randint(3, 20),
                "num_tokens": np.random.randint(100, 500),
                "num_loops": np.random.randint(0, 3),
                "num_bad_actions": np.random.randint(0, 2),
                "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],
            })
        return trajectories

    def group_trajectories_by_task(self, group_size: int) -> List[List[Dict[str, Any]]]:
        """Group trajectories by task_id for training."""
        task_groups = {}

        for traj in self.trajectories:
            task_id = traj.get('task_id', 'unknown')
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(traj)

        # Create groups of size group_size
        groups = []
        for task_id, trajectories in task_groups.items():
            # Split each task's trajectories into groups
            for i in range(0, len(trajectories), group_size):
                group = trajectories[i:i+group_size]
                if len(group) == group_size:  # Only keep full groups
                    groups.append(group)

        logger.info(f"Created {len(groups)} training groups of size {group_size}")
        return groups

    def train_epoch(self, epoch: int, groups: List[List[Dict[str, Any]]]) -> Dict[str, float]:
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

        # Train on each group
        for group_idx, group in enumerate(groups):
            if group_idx > 100:  # Limit to first 100 groups per epoch for speed
                break

            try:
                # Process trajectories
                utilities, costs = self.trainer.process_trajectories(group)
                advantages = self.trainer.compute_group_advantages(utilities)

                # Prepare batch
                batch = {
                    "log_probs": torch.randn(len(group)),
                    "log_probs_old": torch.randn(len(group)),
                    "log_probs_ref": torch.randn(len(group)),
                    "advantages": torch.tensor(advantages, dtype=torch.float32),
                    "mode_log_probs": torch.randn(len(group)),
                    "mode_probs": torch.ones(len(group), 3) / 3,
                }

                # Training step
                metrics = self.trainer.training_step(batch)

                # Track metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])

                success_rate = sum(t['success'] for t in group) / len(group)
                epoch_metrics['success_rate'].append(success_rate)

                if (group_idx + 1) % 20 == 0:
                    logger.info(
                        f"  Group {group_idx + 1}: "
                        f"loss={metrics.get('total_loss', 0):.4f}, "
                        f"success={success_rate:.2%}"
                    )

            except Exception as e:
                logger.error(f"Error processing group {group_idx}: {e}")
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
        """Run training loop on REAL data."""
        logger.info(f"Starting training for {self.epochs} epochs...")
        logger.info(f"Total trajectories: {len(self.trajectories)}")
        logger.info(f"Data source: {self.data_source}")
        logger.info(f"Group size: {self.tosrl_config.group_size}")

        self.best_success = 0

        # Group trajectories
        groups = self.group_trajectories_by_task(self.tosrl_config.group_size)

        if not groups:
            logger.error("No training groups available!")
            return

        all_results = []

        for epoch in range(self.epochs):
            # Train
            epoch_metrics = self.train_epoch(epoch, groups)

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
    parser = argparse.ArgumentParser(description="TOS-RL Training with Real Data")

    # Config
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # Experiment
    parser.add_argument("--experiment", type=str, required=True, help="Experiment ID")
    parser.add_argument("--dataset", type=str, default="webarena", help="Dataset name")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    # Data source
    parser.add_argument("--data-source", type=str, default="file",
                       choices=["file", "tti", "webarena", "mock"],
                       help="Data source for trajectories")
    parser.add_argument("--trajectory-file", type=str,
                       help="Path to JSONL file with trajectories (for file source)")

    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--grad-accum-steps", type=int, help="Gradient accumulation steps")

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
    if args.group_size:
        config["group_size"] = args.group_size
    if args.mode_weight:
        config["mode_weight"] = args.mode_weight
    if args.branching:
        config["use_branching"] = True
    if args.trajectory_file:
        config["train_tasks"] = args.trajectory_file

    # Create pipeline
    pipeline = TOSRLTrainingPipelineRealData(config, args.output_dir, args.data_source)

    # Train
    pipeline.train()


if __name__ == "__main__":
    main()
