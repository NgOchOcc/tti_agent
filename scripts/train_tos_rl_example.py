#!/usr/bin/env python3
"""
Example training script for TOS-RL on WebArena and WebVoyager.

This demonstrates the full TOS-RL training pipeline:
1. Collect trajectories with randomized cost coefficients
2. Group trajectories and compute relative advantages
3. Optional: Generate prefix-level counterfactual branches
4. Training step using GRPO
5. Evaluate on validation set
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tti.tos_rl import (
    TOSRLTrainer, TOSRLConfig,
    CostAwareUtility, batch_compute_utilities,
    BudgetState, TokenModeFormatter,
    PrefixBrancher, BranchCollector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TOSRLTrainingPipeline:
    """Complete training pipeline for TOS-RL."""

    def __init__(
        self,
        config: TOSRLConfig,
        num_epochs: int = 10,
        batch_tasks_per_epoch: int = 32,
    ):
        """
        Initialize pipeline.

        Args:
            config: TOS-RL configuration
            num_epochs: Number of training epochs
            batch_tasks_per_epoch: Tasks per epoch
        """
        self.config = config
        self.num_epochs = num_epochs
        self.batch_tasks_per_epoch = batch_tasks_per_epoch

        # Initialize trainer
        self.trainer = TOSRLTrainer(config)

        # Initialize branching components
        if config.use_branching:
            self.prefix_brancher = PrefixBrancher(
                max_prefixes_per_trajectory=config.max_prefixes_per_task,
                branching_factor=config.branches_per_prefix,
            )
            self.branch_collector = BranchCollector()

        logger.info(f"Initialized TOS-RL pipeline with config: {config}")

    def generate_synthetic_trajectories(
        self,
        num_trajectories: int = 128,
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic trajectories for demo.

        In practice, these would come from running the agent.

        Args:
            num_trajectories: Number of trajectories to generate

        Returns:
            List of trajectory dicts
        """
        trajectories = []

        for i in range(num_trajectories):
            # Randomly succeed/fail
            success = 1 if np.random.rand() > 0.4 else 0

            # Trajectory length varies
            num_steps = np.random.randint(3, 20)
            num_tokens = num_steps * np.random.randint(50, 200)

            # Some trajectories have loops
            num_loops = max(0, np.random.poisson(1) - 2)
            num_bad = max(0, np.random.poisson(0.5) - 1)

            # Generate mode sequence
            modes = []
            for _ in range(num_steps):
                mode = np.random.choice(
                    ["THINK", "OBSERVE"],
                    p=[0.3, 0.7]
                )
                modes.append(mode)
            modes.append("ANSWER")

            trajectory = {
                "task_id": f"task_{i}",
                "success": success,
                "num_steps": num_steps,
                "num_tokens": num_tokens,
                "num_loops": num_loops,
                "num_bad_actions": num_bad,
                "modes": modes,
                "total_tokens": num_tokens,
            }

            trajectories.append(trajectory)

        return trajectories

    def group_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
        group_size: int = 4,
    ) -> List[List[Dict[str, Any]]]:
        """
        Group trajectories (K per task).

        In practice, these come from running K rollouts of the same task.

        Args:
            trajectories: Flat list of trajectories
            group_size: K (group size)

        Returns:
            List of groups (each group is [group_size] trajectories)
        """
        # For demo, just chunk
        groups = []
        for i in range(0, len(trajectories), group_size):
            group = trajectories[i:i + group_size]
            if len(group) == group_size:  # Only full groups
                groups.append(group)

        return groups

    def prepare_training_batch(
        self,
        group: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a training batch from a group of trajectories.

        Args:
            group: List of [group_size] trajectories from same task

        Returns:
            Batch dict ready for training_step
        """
        # Compute utilities with randomized costs
        utilities, costs = self.trainer.process_trajectories(group)

        # Compute group advantages
        advantages = self.trainer.compute_group_advantages(utilities)

        # Convert to tensors
        log_probs = torch.randn(len(group))  # Placeholder
        log_probs_old = torch.randn(len(group))  # Placeholder
        advantages = torch.tensor(advantages, dtype=torch.float32)

        mode_log_probs = torch.randn(len(group))  # Placeholder
        mode_probs = torch.ones(len(group), 3) / 3  # Uniform distribution

        batch = {
            "log_probs": log_probs,
            "log_probs_old": log_probs_old,
            "advantages": advantages,
            "mode_log_probs": mode_log_probs,
            "mode_probs": mode_probs,
        }

        return batch

    def train_epoch(
        self,
        epoch: int,
        trajectories: List[Dict[str, Any]],
        val_trajectories: List[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Epoch number
            trajectories: Training trajectories
            val_trajectories: Optional validation trajectories

        Returns:
            Dictionary of metrics
        """
        logger.info(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")

        # Group trajectories
        groups = self.group_trajectories(trajectories, self.config.group_size)

        epoch_metrics = {
            "total_loss": [],
            "policy_loss": [],
            "mean_advantage": [],
        }

        num_batches = len(groups)

        for batch_idx, group in enumerate(groups):
            # Prepare batch
            batch = self.prepare_training_batch(group)

            # Training step
            metrics = self.trainer.training_step(batch)

            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])

            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = np.mean(epoch_metrics["total_loss"][-self.config.log_interval:])
                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches} - "
                    f"Avg Loss: {avg_loss:.4f}"
                )

        # Compute epoch statistics
        epoch_stats = {
            f"train/{k}": np.mean(v) if v else 0
            for k, v in epoch_metrics.items()
        }

        # Validation
        if val_trajectories:
            val_metrics = self.trainer.evaluate(val_trajectories)
            epoch_stats.update(val_metrics)

            logger.info(
                f"Validation - "
                f"Success: {val_metrics.get('val_success_rate', 0):.2%}, "
                f"Avg Steps: {val_metrics.get('val_mean_steps', 0):.1f}"
            )

        return epoch_stats

    def run_training(
        self,
        num_trajectories_per_epoch: int = 128,
        num_val_trajectories: int = 32,
    ):
        """
        Run complete training loop.

        Args:
            num_trajectories_per_epoch: Trajectories per epoch
            num_val_trajectories: Validation trajectories
        """
        logger.info("Starting TOS-RL training...")

        # Generate validation set
        val_trajectories = self.generate_synthetic_trajectories(num_val_trajectories)

        # Training loop
        for epoch in range(self.num_epochs):
            # Generate training trajectories
            trajectories = self.generate_synthetic_trajectories(
                num_trajectories_per_epoch
            )

            # Train epoch
            metrics = self.train_epoch(epoch, trajectories, val_trajectories)

            logger.info(f"Epoch {epoch + 1} metrics: {metrics}")

            # Save checkpoint (example)
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"checkpoints/tos_rl_epoch_{epoch + 1}.pt"
                self.trainer.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        logger.info("Training completed!")

    def demonstrate_inference(self):
        """
        Demonstrate inference with TOS-RL.
        """
        from tti.tos_rl import TOSRLInference

        logger.info("\n=== Inference Demo ===")

        inference = TOSRLInference(max_steps=10, max_tokens=1024)

        # Mock LLM function
        def mock_llm(prompt: str) -> str:
            """Mock LLM that just returns a random mode."""
            import random
            modes = ["[THINK]", "[OBSERVE]", "[ANSWER]"]
            mode = random.choice(modes)
            content = "dummy content"
            return f"{mode} {content}"

        # Run episode
        result = inference.run_episode(
            task="Find the price of the laptop",
            llm_fn=mock_llm,
            initial_history="User: What's the price of the laptop?\nAssistant: I'll search for it.",
        )

        logger.info(f"Inference result: {result}")

        return result


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train TOS-RL")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--batch-tasks", type=int, default=32, help="Tasks per epoch"
    )
    parser.add_argument(
        "--use-branching", action="store_true", help="Enable prefix branching"
    )
    parser.add_argument(
        "--demo-only", action="store_true", help="Only run inference demo"
    )

    args = parser.parse_args()

    # Configure
    config = TOSRLConfig(
        group_size=4,
        max_prefixes_per_task=3,
        branches_per_prefix=2,
        learning_rate=1e-5,
        mode_token_weight=2.0,
        use_branching=args.use_branching,
        device=torch.device("cpu"),
    )

    # Create pipeline
    pipeline = TOSRLTrainingPipeline(
        config,
        num_epochs=args.epochs,
        batch_tasks_per_epoch=args.batch_tasks,
    )

    if args.demo_only:
        # Just show inference
        pipeline.demonstrate_inference()
    else:
        # Full training
        pipeline.run_training(
            num_trajectories_per_epoch=args.batch_tasks,
            num_val_trajectories=16,
        )

        # Inference demo at end
        pipeline.demonstrate_inference()


if __name__ == "__main__":
    main()
