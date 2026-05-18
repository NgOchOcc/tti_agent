"""
Group-Relative Policy Optimization (GRPO) for TOS-RL.

GRPO is a critic-free RL objective that:
1. Samples K trajectories from the same task
2. Computes group-relative advantages
3. Uses PPO-style clipping
4. Optional mode-token emphasis for faster compute learning

Key insight: No external critic network is trained. Advantages come from
relative ranking within the trajectory group.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """Configuration for GRPO optimization."""

    # Clipping
    clip_ratio: float = 0.2
    epsilon: float = 1e-8

    # Regularization
    kl_weight: float = 0.1
    entropy_weight: float = 0.01

    # Mode emphasis
    mode_token_weight: float = 2.0  # Upweight mode token loss
    use_mode_emphasis: bool = True

    # Learning rate scheduling
    learning_rate: float = 1e-5
    warmup_steps: int = 1000


class GroupRelativePolicyOptimization:
    """
    Group-relative policy optimization for LLM training.

    Key formula: For K trajectories from the same task,
    Â_i = (U_i - mean(U_1:K)) / std(U_1:K)

    This provides an advantage signal without a separate value network.
    """

    def __init__(self, config: GRPOConfig):
        """Initialize GRPO."""
        self.config = config

    @staticmethod
    def compute_group_advantages(
        utilities: np.ndarray,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute group-relative advantages.

        Â_i = (U_i - mean(U)) / std(U)

        Args:
            utilities: [K] utilities from K trajectories of same task
            epsilon: Small constant for numerical stability

        Returns:
            advantages: [K] normalized advantages
        """
        mean_utility = np.mean(utilities)
        std_utility = np.std(utilities)

        # Avoid division by zero
        advantages = (utilities - mean_utility) / (std_utility + epsilon)

        return advantages

    @staticmethod
    def compute_importance_ratio(
        log_probs_new: torch.Tensor,
        log_probs_old: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance sampling ratio.

        ρ = exp(log π_new - log π_old) = π_new / π_old

        Args:
            log_probs_new: Log probs from new policy
            log_probs_old: Log probs from old policy

        Returns:
            ratios: Importance sampling ratios
        """
        return torch.exp(log_probs_new - log_probs_old)


class GRPOLoss:
    """
    Compute GRPO loss with optional mode-token emphasis.
    """

    def __init__(self, config: GRPOConfig):
        """Initialize loss computation."""
        self.config = config

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,  # [batch]
        log_probs_old: torch.Tensor,  # [batch]
        advantages: torch.Tensor,  # [batch]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO-style clipped policy loss.

        L = -1/N Σ min(ρ_t * A_t, clip(ρ_t, 1±ε) * A_t)

        Args:
            log_probs: Log probabilities from new policy
            log_probs_old: Log probabilities from old policy
            advantages: Advantages for each trajectory

        Returns:
            (loss, metrics)
        """
        # Compute importance ratios
        ratios = torch.exp(log_probs - log_probs_old)

        # Clipped objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(
            ratios,
            1 - self.config.clip_ratio,
            1 + self.config.clip_ratio
        ) * advantages

        policy_loss = -torch.mean(torch.min(surr1, surr2))

        metrics = {
            "policy_loss": policy_loss.item(),
            "mean_ratio": ratios.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "clipped_fraction": (
                (torch.abs(ratios - 1) > self.config.clip_ratio).float().mean().item()
            ),
        }

        return policy_loss, metrics

    def compute_mode_emphasis_loss(
        self,
        mode_log_probs: torch.Tensor,  # [batch] log probs of mode tokens
        advantages: torch.Tensor,  # [batch] trajectory advantages
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute mode-token emphasis loss.

        Upweights learning of THINK/OBSERVE/ANSWER decisions.

        L_mode = -α_m Σ log π(m_t) * Â_i

        Args:
            mode_log_probs: Log probs of mode tokens
            advantages: Trajectory advantages

        Returns:
            (loss, metrics)
        """
        if not self.config.use_mode_emphasis:
            return torch.tensor(0.0, device=mode_log_probs.device), {}

        mode_loss = -self.config.mode_token_weight * torch.mean(
            mode_log_probs * advantages
        )

        return mode_loss, {
            "mode_loss": mode_loss.item(),
        }

    def compute_kl_penalty(
        self,
        log_probs_new: torch.Tensor,
        log_probs_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty against reference policy.

        KL = E[log π_new - log π_ref]

        Args:
            log_probs_new: Log probs from new policy
            log_probs_ref: Log probs from reference policy

        Returns:
            KL penalty
        """
        kl_divergence = log_probs_new - log_probs_ref
        return self.config.kl_weight * torch.mean(kl_divergence)

    def compute_entropy_bonus(
        self,
        log_probs: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute entropy regularization bonus.

        Encourages exploration by rewarding high-entropy policies.

        Args:
            log_probs: Log probabilities (for sequence entropy)
            probs: Optional probabilities over modes

        Returns:
            Entropy bonus
        """
        if probs is not None:
            # Entropy over modes: -Σ p_m log p_m
            entropy = -torch.sum(probs * torch.log(probs + self.config.epsilon), dim=-1)
            return self.config.entropy_weight * torch.mean(entropy)
        else:
            # Sequence entropy
            entropy = -torch.mean(log_probs)
            return self.config.entropy_weight * entropy

    def compute_total_loss(
        self,
        log_probs: torch.Tensor,
        log_probs_old: torch.Tensor,
        log_probs_ref: torch.Tensor,
        advantages: torch.Tensor,
        mode_log_probs: Optional[torch.Tensor] = None,
        mode_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total GRPO loss.

        L_total = L_policy + L_mode + L_kl + L_entropy

        Args:
            log_probs: [batch] new policy log probs
            log_probs_old: [batch] old policy log probs
            log_probs_ref: [batch] reference policy log probs
            advantages: [batch] group-relative advantages
            mode_log_probs: [batch] log probs of mode tokens (optional)
            mode_probs: [batch, 3] mode probabilities for entropy (optional)

        Returns:
            (total_loss, metrics)
        """
        metrics = {}

        # Policy loss (main objective)
        policy_loss, policy_metrics = self.compute_policy_loss(
            log_probs, log_probs_old, advantages
        )
        metrics.update(policy_metrics)

        # Mode emphasis (optional)
        mode_loss = torch.tensor(0.0, device=log_probs.device)
        if mode_log_probs is not None:
            mode_loss, mode_metrics = self.compute_mode_emphasis_loss(
                mode_log_probs, advantages
            )
            metrics.update(mode_metrics)

        # KL penalty
        kl_loss = self.compute_kl_penalty(log_probs, log_probs_ref)
        metrics["kl_loss"] = kl_loss.item()

        # Entropy bonus
        entropy_bonus = self.compute_entropy_bonus(log_probs, mode_probs)
        metrics["entropy_bonus"] = entropy_bonus.item()

        # Total loss
        total_loss = policy_loss + mode_loss + kl_loss - entropy_bonus

        return total_loss, metrics
