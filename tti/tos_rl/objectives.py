"""
Cost-aware utility objectives for TOS-RL.

The LLM learns to optimize:
U(τ) = R_task(τ) - λ_env·N_env - λ_tok·N_tok - λ_loop·N_loop - λ_bad·N_bad

This creates implicit value-of-information learning:
- OBSERVE gets positive gradient when new observation improves utility
- THINK gets positive gradient when reasoning improves utility
- ANSWER gets positive gradient when early stopping preserves success
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class CostAwareUtility:
    """Container for cost-aware utility computation."""

    task_reward: float  # R_task(τ): 0/1 for success, or graded score
    num_env_steps: int  # N_env: number of environment interactions
    num_tokens: int  # N_tok: total tokens generated
    num_loops: int  # N_loop: repeated states or actions
    num_bad_actions: int  # N_bad: invalid or irreversible actions

    def compute(
        self,
        lambda_env: float = 0.01,
        lambda_tok: float = 0.001,
        lambda_loop: float = 0.1,
        lambda_bad: float = 0.2,
    ) -> float:
        """
        Compute cost-aware utility.

        U(τ) = R_task - λ_env·N_env - λ_tok·N_tok - λ_loop·N_loop - λ_bad·N_bad

        Args:
            lambda_env: Cost per environment step
            lambda_tok: Cost per token
            lambda_loop: Cost per loop/repeated action
            lambda_bad: Cost per bad/invalid action

        Returns:
            Utility value
        """
        cost = (
            lambda_env * self.num_env_steps +
            lambda_tok * self.num_tokens +
            lambda_loop * self.num_loops +
            lambda_bad * self.num_bad_actions
        )

        utility = self.task_reward - cost
        return utility

    def __repr__(self) -> str:
        return (
            f"U = {self.task_reward:.2f} - "
            f"(0.01×{self.num_env_steps} + 0.001×{self.num_tokens} + "
            f"0.1×{self.num_loops} + 0.2×{self.num_bad_actions})"
        )


def compute_cost_aware_utility(
    task_reward: float,
    num_env_steps: int,
    num_tokens: int,
    num_loops: int = 0,
    num_bad_actions: int = 0,
    cost_coefficients: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute cost-aware utility for a single trajectory.

    Args:
        task_reward: Task success/failure (0 or 1)
        num_env_steps: Number of browser/environment interactions
        num_tokens: Total tokens generated
        num_loops: Number of repeated actions/states
        num_bad_actions: Number of invalid/bad actions
        cost_coefficients: Optional dict with keys:
            - "env": λ_env (default 0.01)
            - "tok": λ_tok (default 0.001)
            - "loop": λ_loop (default 0.1)
            - "bad": λ_bad (default 0.2)

    Returns:
        Utility value
    """
    if cost_coefficients is None:
        cost_coefficients = {}

    lambda_env = cost_coefficients.get("env", 0.01)
    lambda_tok = cost_coefficients.get("tok", 0.001)
    lambda_loop = cost_coefficients.get("loop", 0.1)
    lambda_bad = cost_coefficients.get("bad", 0.2)

    utility = CostAwareUtility(
        task_reward=task_reward,
        num_env_steps=num_env_steps,
        num_tokens=num_tokens,
        num_loops=num_loops,
        num_bad_actions=num_bad_actions,
    )

    return utility.compute(lambda_env, lambda_tok, lambda_loop, lambda_bad)


def batch_compute_utilities(
    trajectories: List[Dict[str, Any]],
    cost_coefficients: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute utilities for a batch of trajectories.

    Args:
        trajectories: List of trajectory dicts with:
            - "success": 0 or 1 (or task reward)
            - "num_steps": number of environment steps
            - "num_tokens": tokens generated
            - "num_loops": repeated actions
            - "num_bad_actions": invalid actions
        cost_coefficients: Optional cost dict

    Returns:
        utilities: [batch_size] array of utility values
    """
    utilities = []

    for traj in trajectories:
        reward = float(traj.get("success", 0))
        num_steps = int(traj.get("num_steps", 0))
        num_tokens = int(traj.get("num_tokens", 0))
        num_loops = int(traj.get("num_loops", 0))
        num_bad = int(traj.get("num_bad_actions", 0))

        utility = compute_cost_aware_utility(
            reward, num_steps, num_tokens, num_loops, num_bad,
            cost_coefficients
        )

        utilities.append(utility)

    return np.array(utilities)


def normalize_utilities(
    utilities: np.ndarray,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize utilities to zero mean and unit variance.

    Args:
        utilities: [batch_size] utility values
        epsilon: Small constant for numerical stability

    Returns:
        (normalized_utilities, mean, std)
    """
    mean = np.mean(utilities)
    std = np.std(utilities)

    normalized = (utilities - mean) / (std + epsilon)

    return normalized, mean, std


class UtilityTracker:
    """
    Tracks utility statistics during training.
    """

    def __init__(self):
        """Initialize tracker."""
        self.utilities = []
        self.rewards = []
        self.costs = []

    def record(self, utility: float, reward: float, cost: float):
        """Record a trajectory's utility and components."""
        self.utilities.append(utility)
        self.rewards.append(reward)
        self.costs.append(cost)

    def get_stats(self) -> Dict[str, float]:
        """Get statistics."""
        if not self.utilities:
            return {}

        return {
            "mean_utility": float(np.mean(self.utilities)),
            "std_utility": float(np.std(self.utilities)),
            "min_utility": float(np.min(self.utilities)),
            "max_utility": float(np.max(self.utilities)),
            "mean_reward": float(np.mean(self.rewards)),
            "mean_cost": float(np.mean(self.costs)),
            "success_rate": float(np.mean([1 if r > 0 else 0 for r in self.rewards])),
        }

    def reset(self):
        """Reset tracker."""
        self.utilities = []
        self.rewards = []
        self.costs = []
