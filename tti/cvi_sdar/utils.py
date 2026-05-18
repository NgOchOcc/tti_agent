"""
Utility functions for CVI-SDAR training and inference.

Includes:
- CVI gap computation
- Budget tracking and management
- Trajectory utilities
- Cost-aware return computation
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
import copy


@dataclass
class Budget:
    """Budget state for agent execution."""

    env_interactions: float = 100  # Remaining environment interactions
    tokens: float = 4096  # Remaining tokens
    verifier_calls: float = 10  # Remaining verifier calls
    risk_tolerance: float = 1.0  # Risk tolerance (0-1)

    def copy(self) -> "Budget":
        """Create a copy of this budget."""
        return copy.deepcopy(self)

    def as_dict(self) -> Dict[str, float]:
        """Return budget as dictionary."""
        return {
            "env_interactions": self.env_interactions,
            "tokens": self.tokens,
            "verifier_calls": self.verifier_calls,
            "risk_tolerance": self.risk_tolerance,
        }

    def __repr__(self) -> str:
        return (
            f"Budget(env={self.env_interactions:.0f}, "
            f"tok={self.tokens:.0f}, "
            f"ver={self.verifier_calls:.0f}, "
            f"risk={self.risk_tolerance:.2f})"
        )


@dataclass
class CostMetrics:
    """Metrics for cost tracking."""

    num_env_steps: int = 0
    num_tokens_generated: int = 0
    num_verifier_calls: int = 0
    num_loops: int = 0
    num_invalid_actions: int = 0
    num_recoveries: int = 0

    def to_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Convert metrics to tensor for batching."""
        return torch.tensor([
            self.num_env_steps,
            self.num_tokens_generated,
            self.num_verifier_calls,
            self.num_loops,
            self.num_invalid_actions,
            self.num_recoveries,
        ], dtype=torch.float32, device=device)


def compute_cost_aware_return(
    rewards: np.ndarray,
    cost_metrics: List[CostMetrics],
    cost_coefficients: Dict[str, float],
    gamma: float = 0.99,
    normalize_cost: bool = True,
) -> np.ndarray:
    """
    Compute cost-aware returns (utilities).

    U(τ) = R(τ) - λ_env * N_env - λ_tok * N_tok - λ_ver * N_ver - λ_risk * C_risk

    Args:
        rewards: [batch_size] or [seq_len] task rewards (success/failure)
        cost_metrics: List of CostMetrics for each trajectory
        cost_coefficients: Dictionary with keys:
            - "env": λ_env (cost per environment step)
            - "tok": λ_tok (cost per token)
            - "ver": λ_ver (cost per verifier call)
            - "loop": λ_loop (penalty per loop)
            - "invalid": λ_invalid (penalty per invalid action)
        gamma: Discount factor for cumulative discounted costs
        normalize_cost: If True, normalize costs by trajectory length

    Returns:
        utilities: [batch_size] or [seq_len] cost-aware utilities
    """
    # Extract cost coefficients
    cost_env = cost_coefficients.get("env", 0.01)
    cost_tok = cost_coefficients.get("tok", 0.001)
    cost_ver = cost_coefficients.get("ver", 0.05)
    cost_loop = cost_coefficients.get("loop", 0.1)
    cost_invalid = cost_coefficients.get("invalid", 0.2)

    utilities = []

    for i, (reward, metrics) in enumerate(zip(rewards, cost_metrics)):
        # Compute total cost
        total_cost = (
            cost_env * metrics.num_env_steps +
            cost_tok * metrics.num_tokens_generated +
            cost_ver * metrics.num_verifier_calls +
            cost_loop * metrics.num_loops +
            cost_invalid * metrics.num_invalid_actions
        )

        # Cost-aware utility
        utility = reward - total_cost

        utilities.append(utility)

    return np.array(utilities)


def compute_cvi_gap(
    q_values: torch.Tensor,
    v_ans_values: torch.Tensor,
    modes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Counterfactual Value-of-Interaction (CVI) gaps.

    Δ_m(h_t, b_t) = Q(h_t, b_t, m) - V_ans(h_t, b_t)

    Args:
        q_values: [batch_size, num_modes] Q-function values
        v_ans_values: [batch_size] Answer value function
        modes: [batch_size] optional mode indices to extract specific Q values

    Returns:
        cvi_gaps: [batch_size, num_modes] or [batch_size] if modes provided
    """
    batch_size = q_values.shape[0]
    v_ans_expanded = v_ans_values.unsqueeze(-1)  # [batch_size, 1]

    # Compute gaps for all modes
    cvi_gaps = q_values - v_ans_expanded  # [batch_size, num_modes]

    if modes is not None:
        # Extract gaps only for the taken modes
        cvi_gaps = cvi_gaps[torch.arange(batch_size), modes]  # [batch_size]

    return cvi_gaps


def select_mode_by_cvi(
    q_values: torch.Tensor,
    v_ans_values: torch.Tensor,
    delta_threshold: float = 0.0,
    exclude_answer: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select the best mode based on CVI gap.

    m_t* = argmax_{m ≠ ANSWER} Δ_m(h_t, b_t)

    If max Δ_m ≤ delta_threshold, return ANSWER mode.

    Args:
        q_values: [batch_size, num_modes] Q-values
        v_ans_values: [batch_size] V_ans values
        delta_threshold: Threshold for continuing (default 0.0)
        exclude_answer: If True, never select ANSWER mode (for training only)

    Returns:
        selected_modes: [batch_size] selected mode indices
        cvi_gaps: [batch_size] CVI gap values for selected modes
    """
    # Compute CVI gaps
    cvi_gaps = compute_cvi_gap(q_values, v_ans_values)  # [batch_size, num_modes]

    batch_size = q_values.shape[0]
    num_modes = q_values.shape[1]

    if exclude_answer:
        # Exclude ANSWER mode (index num_modes - 1)
        q_values_excluding_answer = q_values[:, :-1]  # [batch_size, num_modes-1]
        cvi_gaps_excluding_answer = cvi_gaps[:, :-1]

        # Get best mode
        selected_modes = torch.argmax(cvi_gaps_excluding_answer, dim=-1)  # [batch_size]
        best_gaps = cvi_gaps_excluding_answer.max(dim=-1).values
    else:
        # Include all modes
        selected_modes = torch.argmax(cvi_gaps, dim=-1)  # [batch_size]
        best_gaps = cvi_gaps.max(dim=-1).values

    # Check threshold: if max gap ≤ threshold, select ANSWER (last mode)
    answer_mode = num_modes - 1
    should_answer = best_gaps <= delta_threshold

    selected_modes = torch.where(
        should_answer,
        torch.full_like(selected_modes, answer_mode),
        selected_modes
    )

    # Get the actual CVI gaps for selected modes
    selected_gaps = cvi_gaps[torch.arange(batch_size), selected_modes]

    return selected_modes, selected_gaps


def batch_trajectory_costs(
    trajectories: List[Dict[str, Any]],
    cost_coefficients: Dict[str, float],
) -> np.ndarray:
    """
    Compute costs for a batch of trajectories.

    Args:
        trajectories: List of trajectory dictionaries
        cost_coefficients: Cost coefficients

    Returns:
        costs: [batch_size] cost values
    """
    costs = []

    for traj in trajectories:
        # Extract metrics from trajectory
        num_env_steps = len(traj.get("actions", []))
        num_tokens = traj.get("num_tokens_generated", 0)
        num_verifiers = traj.get("num_verifier_calls", 0)
        num_loops = traj.get("num_loops", 0)
        num_invalid = traj.get("num_invalid_actions", 0)

        # Compute cost
        cost = (
            cost_coefficients.get("env", 0.01) * num_env_steps +
            cost_coefficients.get("tok", 0.001) * num_tokens +
            cost_coefficients.get("ver", 0.05) * num_verifiers +
            cost_coefficients.get("loop", 0.1) * num_loops +
            cost_coefficients.get("invalid", 0.2) * num_invalid
        )

        costs.append(cost)

    return np.array(costs)


def compute_discounted_returns(
    rewards: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """
    Compute discounted cumulative returns from rewards.

    G_t = R_t + γ * R_{t+1} + γ^2 * R_{t+2} + ...

    Args:
        rewards: [seq_len] reward sequence
        gamma: Discount factor

    Returns:
        returns: [seq_len] discounted returns
    """
    returns = np.zeros_like(rewards, dtype=np.float32)

    # Backward pass to compute discounted returns
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def normalize_returns(
    returns: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalize returns using running statistics.

    Args:
        returns: [batch_size] or [seq_len] returns
        epsilon: Small constant for numerical stability

    Returns:
        normalized_returns: Normalized returns with zero mean and unit variance
    """
    mean = np.mean(returns)
    std = np.std(returns)

    return (returns - mean) / (std + epsilon)


def compute_advantages(
    returns: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE).

    A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Args:
        returns: [seq_len] returns
        values: [seq_len] value estimates
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: [seq_len] advantage estimates
    """
    advantages = np.zeros_like(returns, dtype=np.float32)
    gae = 0.0

    # Backward pass
    for t in reversed(range(len(returns))):
        if t == len(returns) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        # Temporal difference error
        delta = returns[t] + gamma * next_value - values[t]

        # GAE
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    return advantages


def moving_average(
    values: np.ndarray,
    window_size: int = 10
) -> np.ndarray:
    """
    Compute moving average of values.

    Args:
        values: [seq_len] values
        window_size: Size of moving average window

    Returns:
        averaged: [seq_len] moving averaged values
    """
    averaged = np.convolve(
        values,
        np.ones(window_size) / window_size,
        mode="valid"
    )

    # Pad start with original values
    padding = values[:window_size - 1]
    return np.concatenate([padding, averaged])
