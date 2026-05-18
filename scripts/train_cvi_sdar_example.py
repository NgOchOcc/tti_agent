#!/usr/bin/env python3
"""
Example training script for CVI-SDAR on WebArena and WebVoyager.

This script shows how to integrate CVI-SDAR into the TTI training pipeline.
Modify based on your specific setup and environment.
"""

import torch
import torch.nn as nn
import argparse
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tti.cvi_sdar import (
    CVISARTrainer,
    CVISARConfig,
    PrefixMiner,
    BranchGenerator,
    Budget,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CVISARTrainingPipeline:
    """Complete training pipeline for CVI-SDAR."""

    def __init__(
        self,
        config: CVISARConfig,
        save_dir: Path = Path("./checkpoints/cvi_sdar"),
        log_interval: int = 10,
    ):
        """Initialize the training pipeline."""
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        # Initialize trainer
        self.trainer = CVISARTrainer(config)

        # Initialize mining components
        self.prefix_miner = PrefixMiner(
            sampling_strategy="mixed",
            max_prefixes_per_trajectory=config.prefixes_per_trajectory,
        )

        self.branch_generator = BranchGenerator(
            num_modes=config.num_modes,
            max_branch_length=None,
            use_privileged_budget=True,
        )

        logger.info(f"Initialized CVI-SDAR trainer with config: {config}")

    def prepare_batch(
        self,
        trajectories: List[Dict[str, Any]],
        get_state_embedding: callable,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of trajectories for training.

        Args:
            trajectories: List of trajectories
            get_state_embedding: Function to extract state embeddings

        Returns:
            Batch dictionary with tensors
        """
        batch_data = {
            "state_embeddings": [],
            "modes": [],
            "returns": [],
            "advantages": [],
            "old_log_probs": [],
            "answer_utilities": [],
        }

        for traj in trajectories:
            # Get state embeddings for each step
            for step_idx, _ in enumerate(traj.get("actions", [])):
                emb = get_state_embedding(traj, step_idx)
                if emb is None:
                    continue

                batch_data["state_embeddings"].append(
                    torch.tensor(emb, dtype=torch.float32)
                )
                batch_data["modes"].append(traj["modes"][step_idx])

            # Compute returns and advantages
            rewards = np.array(traj.get("rewards", []))
            gamma = self.config.gamma
            returns = self._compute_returns(rewards, gamma)
            advantages = self._compute_advantages(returns)

            batch_data["returns"].extend(returns)
            batch_data["advantages"].extend(advantages)

            # Placeholder log probs (would come from policy evaluation)
            batch_data["old_log_probs"].extend(
                [np.log(1.0 / self.config.num_modes)] * len(rewards)
            )

        # Convert to tensors
        batch = {
            "state_embeddings": torch.stack(batch_data["state_embeddings"]),
            "modes": torch.tensor(batch_data["modes"], dtype=torch.long),
            "returns": torch.tensor(batch_data["returns"], dtype=torch.float32),
            "advantages": torch.tensor(batch_data["advantages"], dtype=torch.float32),
            "old_log_probs": torch.tensor(batch_data["old_log_probs"], dtype=torch.float32),
            "answer_utilities": torch.tensor(
                batch_data["answer_utilities"], dtype=torch.float32
            ),
        }

        return batch

    def prepare_prefix_data(
        self,
        trajectories: List[Dict[str, Any]],
        get_state_embedding: callable,
    ) -> Dict[str, Any]:
        """
        Prepare prefix and branch data for distillation.

        Args:
            trajectories: List of trajectories
            get_state_embedding: Function to extract state embeddings

        Returns:
            Prefix data dictionary
        """
        teacher_modes = []
        cvi_advantages = []
        state_embeddings_for_distill = []

        for traj in trajectories:
            # Mine prefixes
            prefixes = self.prefix_miner.mine_prefixes(traj)

            for prefix in prefixes:
                # Generate branches
                branches = self.branch_generator.generate_branches(
                    prefix, traj
                )

                if not branches:
                    continue

                # Compute teacher mode
                teacher_mode, _ = self.branch_generator.compute_teacher_mode(
                    branches
                )
                cvi_adv = self.branch_generator.compute_cvi_advantage(
                    branches, teacher_mode
                )

                # Get state embedding for this prefix
                emb = get_state_embedding(traj, prefix.step_index)
                if emb is not None:
                    state_embeddings_for_distill.append(
                        torch.tensor(emb, dtype=torch.float32)
                    )
                    teacher_modes.append(teacher_mode)
                    cvi_advantages.append(cvi_adv)

        prefix_data = {
            "teacher_modes": torch.tensor(teacher_modes, dtype=torch.long),
            "cvi_advantages": torch.tensor(cvi_advantages, dtype=torch.float32),
            "state_embeddings": (
                torch.stack(state_embeddings_for_distill)
                if state_embeddings_for_distill else None
            ),
        }

        return prefix_data

    def train_epoch(
        self,
        train_trajectories: List[Dict[str, Any]],
        get_state_embedding: callable,
        val_trajectories: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_trajectories: Training trajectories
            get_state_embedding: Function to extract state embeddings
            val_trajectories: Optional validation trajectories

        Returns:
            Dictionary of metrics
        """
        epoch_losses = {
            "ppo_loss": [],
            "critic_loss": [],
            "distill_loss": [],
        }

        num_batches = max(1, len(train_trajectories) // self.config.batch_size)

        for batch_idx in range(num_batches):
            # Sample batch
            batch_traj = train_trajectories[
                batch_idx * self.config.batch_size:
                (batch_idx + 1) * self.config.batch_size
            ]

            if not batch_traj:
                continue

            # Prepare batch
            batch = self.prepare_batch(batch_traj, get_state_embedding)

            # Prepare prefix data for distillation
            prefix_data = self.prepare_prefix_data(batch_traj, get_state_embedding)

            # Move batch to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            # Training step
            losses = self.trainer.train_step(batch, prefix_data)

            for key in epoch_losses:
                epoch_losses[key].append(losses.get(key, 0))

            if batch_idx % self.log_interval == 0:
                avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
                logger.info(
                    f"Batch {batch_idx}/{num_batches} - "
                    f"PPO: {avg_losses['ppo_loss']:.4f}, "
                    f"Critic: {avg_losses['critic_loss']:.4f}, "
                    f"Distill: {avg_losses['distill_loss']:.4f}"
                )

        # Compute epoch metrics
        epoch_metrics = {
            f"train/{k}": np.mean(v) for k, v in epoch_losses.items()
        }

        # Validation
        if val_trajectories:
            val_metrics = self.trainer.evaluate(val_trajectories, get_state_embedding)
            epoch_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})

        return epoch_metrics

    def save_checkpoint(self, path: Path, epoch: int):
        """Save training checkpoint."""
        self.trainer.save_checkpoint(str(path))
        logger.info(f"Saved checkpoint at epoch {epoch} to {path}")

    @staticmethod
    def _compute_returns(rewards: np.ndarray, gamma: float) -> List[float]:
        """Compute discounted returns."""
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns.tolist()

    @staticmethod
    def _compute_advantages(returns: np.ndarray) -> List[float]:
        """Compute advantages (baseline is zero for simplicity)."""
        mean = np.mean(returns)
        std = np.std(returns) + 1e-8
        advantages = (returns - mean) / std
        return advantages.tolist()


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train CVI-SDAR on WebArena/WebVoyager")
    parser.add_argument(
        "--benchmark",
        choices=["webarena", "webvoyager"],
        default="webarena",
        help="Which benchmark to train on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("./checkpoints/cvi_sdar"),
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    # Create config
    config = CVISARConfig(
        state_embedding_dim=768,  # Gemma hidden size
        num_modes=6,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )

    # Create pipeline
    pipeline = CVISARTrainingPipeline(config, save_dir=args.save_dir)

    # Placeholder: Load your actual training data
    # In practice, load from WebArena or WebVoyager
    train_trajectories = []  # Load from dataset
    val_trajectories = []    # Load from dataset

    def get_state_embedding(traj: Dict, step: int) -> np.ndarray:
        """Extract state embedding from trajectory."""
        # Placeholder: would extract from model's hidden states
        return np.random.randn(768).astype(np.float32)

    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        metrics = pipeline.train_epoch(
            train_trajectories,
            get_state_embedding,
            val_trajectories,
        )

        logger.info(f"Epoch metrics: {metrics}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = args.save_dir / f"epoch_{epoch + 1}.pt"
            pipeline.save_checkpoint(checkpoint_path, epoch + 1)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
