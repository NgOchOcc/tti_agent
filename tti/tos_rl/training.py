"""
Training loop for TOS-RL.

Key features:
- No external networks (only LLM)
- Group-relative advantage computation
- Mode-token emphasis
- Prefix-level branching for improved credit assignment
- Budget-conditioned training
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .objectives import batch_compute_utilities, normalize_utilities
from .optimization import GRPOLoss, GroupRelativePolicyOptimization, GRPOConfig
from .branching import PrefixBrancher, BranchCollector, Prefix
from .utils import CostCoefficientSchedule, TokenModeFormatter, BudgetState

logger = logging.getLogger(__name__)


@dataclass
class TOSRLConfig:
    """Configuration for TOS-RL training."""

    # Cost coefficients
    lambda_env_range: Tuple[float, float] = (0.001, 0.1)
    lambda_tok_range: Tuple[float, float] = (0.0001, 0.01)
    lambda_loop_range: Tuple[float, float] = (0.01, 0.5)
    lambda_bad_range: Tuple[float, float] = (0.1, 1.0)

    # RL parameters
    group_size: int = 4  # K trajectories per task
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # GRPO parameters
    clip_ratio: float = 0.2
    kl_weight: float = 0.1
    entropy_weight: float = 0.01
    mode_token_weight: float = 2.0

    # Branching parameters
    max_prefixes_per_task: int = 3
    branches_per_prefix: int = 3
    use_branching: bool = True

    # Training
    num_epochs: int = 10
    save_interval: int = 1000
    log_interval: int = 100

    device: torch.device = torch.device("cpu")


class TOSRLTrainer:
    """
    LLM-only trainer for TOS-RL.

    Trains a single language model to decide when to THINK, OBSERVE, or ANSWER.
    """

    def __init__(
        self,
        config: TOSRLConfig,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = config.device
        self.global_step = 0

        # Cost schedule
        self.cost_schedule = CostCoefficientSchedule(
            lambda_env_range=config.lambda_env_range,
            lambda_tok_range=config.lambda_tok_range,
            lambda_loop_range=config.lambda_loop_range,
            lambda_bad_range=config.lambda_bad_range,
        )

        # GRPO loss
        grpo_config = GRPOConfig(
            clip_ratio=config.clip_ratio,
            kl_weight=config.kl_weight,
            entropy_weight=config.entropy_weight,
            mode_token_weight=config.mode_token_weight,
            use_mode_emphasis=True,
        )
        self.grpo_loss_fn = GRPOLoss(grpo_config)

        # Prefix brancher
        self.prefix_brancher = PrefixBrancher(
            max_prefixes_per_trajectory=config.max_prefixes_per_task,
            branching_factor=config.branches_per_prefix,
        )

        logger.info(f"Initialized TOS-RL trainer with config: {config}")

    def process_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process a batch of trajectories.

        Args:
            trajectories: List of trajectory dicts with:
                - "success": Task success (0 or 1)
                - "num_steps": Number of environment interactions
                - "num_tokens": Tokens generated
                - "num_loops": Repeated actions/states
                - "num_bad_actions": Invalid actions

        Returns:
            (utilities, cost_coefficients)
        """
        # Sample cost coefficients for this batch
        costs = self.cost_schedule.sample()

        # Compute utilities
        utilities = batch_compute_utilities(trajectories, costs)

        return utilities, costs

    def compute_group_advantages(
        self,
        group_utilities: np.ndarray,
    ) -> np.ndarray:
        """
        Compute advantages for a group of trajectories.

        Â_i = (U_i - mean(U)) / std(U)

        Args:
            group_utilities: [K] utilities from K trajectories

        Returns:
            advantages: [K] normalized advantages
        """
        return GroupRelativePolicyOptimization.compute_group_advantages(
            group_utilities
        )

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        log_probs_ref: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch dict with:
                - "log_probs": [batch_size] log probs
                - "log_probs_old": [batch_size] old log probs
                - "advantages": [batch_size] advantages
                - "mode_log_probs": [batch_size] mode log probs (optional)
                - "mode_probs": [batch_size, 3] mode probs (optional)
            log_probs_ref: Reference policy log probs (optional)

        Returns:
            Dictionary of loss values and metrics
        """
        self.global_step += 1

        log_probs = batch["log_probs"].to(self.device)
        log_probs_old = batch["log_probs_old"].to(self.device)
        advantages = batch["advantages"].to(self.device).float()

        mode_log_probs = batch.get("mode_log_probs")
        if mode_log_probs is not None:
            mode_log_probs = mode_log_probs.to(self.device)

        mode_probs = batch.get("mode_probs")
        if mode_probs is not None:
            mode_probs = mode_probs.to(self.device)

        if log_probs_ref is None:
            log_probs_ref = log_probs_old.detach()

        # Compute total loss
        loss, metrics = self.grpo_loss_fn.compute_total_loss(
            log_probs=log_probs,
            log_probs_old=log_probs_old,
            log_probs_ref=log_probs_ref,
            advantages=advantages,
            mode_log_probs=mode_log_probs,
            mode_probs=mode_probs,
        )

        metrics["total_loss"] = loss.item()
        metrics["mean_advantage"] = advantages.mean().item()

        return metrics

    def branch_training_step(
        self,
        branch_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Training step using prefix-level branches.

        Args:
            branch_batch: Batch from counterfactual branches with:
                - "branch_log_probs": Log probs of branch tokens
                - "branch_advantages": Advantages within each prefix
                - "branch_mode_log_probs": Log probs of mode tokens

        Returns:
            Dictionary of metrics
        """
        log_probs = branch_batch["branch_log_probs"].to(self.device)
        advantages = branch_batch["branch_advantages"].to(self.device).float()

        mode_log_probs = branch_batch.get("branch_mode_log_probs")
        if mode_log_probs is not None:
            mode_log_probs = mode_log_probs.to(self.device)

        # Simple loss: -E[log π(tokens) * A]
        policy_loss = -torch.mean(log_probs * advantages)

        metrics = {
            "branch_loss": policy_loss.item(),
            "branch_mean_advantage": advantages.mean().item(),
        }

        if mode_log_probs is not None:
            mode_loss = -self.config.mode_token_weight * torch.mean(
                mode_log_probs * advantages
            )
            metrics["branch_mode_loss"] = mode_loss.item()
            policy_loss = policy_loss + mode_loss

        return metrics

    def evaluate(
        self,
        val_trajectories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate on validation trajectories.

        Args:
            val_trajectories: List of validation trajectories

        Returns:
            Dictionary of evaluation metrics
        """
        if not val_trajectories:
            return {}

        utilities, costs = self.process_trajectories(val_trajectories)

        success_rate = np.mean([
            float(t.get("success", 0))
            for t in val_trajectories
        ])

        avg_steps = np.mean([
            t.get("num_steps", 0)
            for t in val_trajectories
        ])

        avg_tokens = np.mean([
            t.get("num_tokens", 0)
            for t in val_trajectories
        ])

        return {
            "val_success_rate": success_rate,
            "val_mean_utility": np.mean(utilities),
            "val_mean_steps": avg_steps,
            "val_mean_tokens": avg_tokens,
        }

    def save_checkpoint(self, path: str):
        """Save checkpoint (placeholder for LLM checkpoint)."""
        logger.info(f"Would save checkpoint to {path}")
        # In practice, save LLM weights here

    def load_checkpoint(self, path: str):
        """Load checkpoint (placeholder for LLM checkpoint)."""
        logger.info(f"Would load checkpoint from {path}")
        # In practice, load LLM weights here
