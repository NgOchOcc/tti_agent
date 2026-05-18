"""
Prefix-level counterfactual branching for TOS-RL.

Improves credit assignment without an external critic:
1. Select informative prefixes from trajectories
2. Generate forced continuations: τ^THINK, τ^OBSERVE, τ^ANSWER
3. Compute branch utilities and relative advantages
4. Update mode tokens using branch advantages

This provides local counterfactual comparisons: "what if we THINKed vs OBSERVEd here?"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random


@dataclass
class Prefix:
    """Represents a trajectory prefix."""

    trajectory_id: str
    step_index: int
    history: Dict[str, Any]  # History up to this point
    budget: Dict[str, float]  # Remaining budget
    is_high_uncertainty: bool = False  # Whether this was a high-uncertainty point


@dataclass
class Branch:
    """Represents a counterfactual branch from a prefix."""

    prefix: Prefix
    forced_mode: str  # "THINK", "OBSERVE", or "ANSWER"
    continuation_tokens: List[int]  # Generated tokens from this prefix
    branch_utility: float  # Utility of this branch
    num_steps_added: int = 0
    num_tokens_added: int = 0


class PrefixBrancher:
    """
    Selects informative prefixes and generates counterfactual branches.

    Strategy: Sample prefixes from:
    - High-uncertainty decision points
    - Mixed-outcome trajectories (some succeed, some fail)
    - Points before critical actions
    """

    def __init__(
        self,
        max_prefixes_per_trajectory: int = 3,
        branching_factor: int = 3,  # Number of continuations per mode
        random_seed: int = 42,
    ):
        """
        Args:
            max_prefixes_per_trajectory: Max prefixes to select per trajectory
            branching_factor: Number of continuations per mode
            random_seed: Random seed for reproducibility
        """
        self.max_prefixes_per_trajectory = max_prefixes_per_trajectory
        self.branching_factor = branching_factor
        random.seed(random_seed)

    def select_prefixes(
        self,
        trajectory: Dict[str, Any],
        group_utilities: Optional[np.ndarray] = None,
        mixed_outcomes: bool = False,
    ) -> List[Prefix]:
        """
        Select informative prefixes from a trajectory.

        Args:
            trajectory: Trajectory dict with:
                - "history": List of observations/actions
                - "modes": List of mode tokens
                - "success": Final success
                - "num_steps": Number of steps
            group_utilities: [K] utilities from K trajectories (for uncertainty)
            mixed_outcomes: If True, prefer prefixes from mixed-outcome groups

        Returns:
            List of Prefix objects
        """
        prefixes = []
        num_steps = trajectory.get("num_steps", 0)

        if num_steps == 0:
            return prefixes

        # Strategy: Select prefixes at regular intervals
        if self.max_prefixes_per_trajectory >= num_steps:
            selected_indices = list(range(num_steps))
        else:
            # Divide into segments
            segment_size = num_steps / self.max_prefixes_per_trajectory
            selected_indices = [
                int((i + 0.5) * segment_size)
                for i in range(self.max_prefixes_per_trajectory)
            ]

        # Create Prefix objects
        for step_idx in selected_indices:
            # Estimate remaining budget
            remaining_budget = {
                "env": max(0, 30 - step_idx),  # Assuming max 30 steps
                "tokens": max(0, 4096 - trajectory.get("total_tokens", 0)),
                "time": 1.0,
            }

            # Check if high uncertainty (relevant if group_utilities available)
            is_uncertain = False
            if group_utilities is not None and len(group_utilities) > 1:
                std = np.std(group_utilities)
                is_uncertain = std > np.mean(group_utilities) * 0.1

            prefix = Prefix(
                trajectory_id=trajectory.get("task_id", "unknown"),
                step_index=step_idx,
                history={
                    "observations": trajectory.get("observations", [])[:step_idx + 1],
                    "modes": trajectory.get("modes", [])[:step_idx],
                    "actions": trajectory.get("actions", [])[:step_idx],
                },
                budget=remaining_budget,
                is_high_uncertainty=is_uncertain,
            )

            prefixes.append(prefix)

        return prefixes[:self.max_prefixes_per_trajectory]


class BranchCollector:
    """
    Collects and manages counterfactual branches.

    In practice, this would:
    1. Force the LLM to emit specific mode tokens
    2. Generate continuations
    3. Evaluate utilities
    4. Store for training
    """

    def __init__(self):
        """Initialize collector."""
        self.branches = []
        self.branch_utilities = {}

    def add_branch(
        self,
        prefix: Prefix,
        mode: str,
        tokens: List[int],
        utility: float,
        num_steps_added: int = 0,
        num_tokens_added: int = 0,
    ):
        """Add a branch to the collection."""
        branch = Branch(
            prefix=prefix,
            forced_mode=mode,
            continuation_tokens=tokens,
            branch_utility=utility,
            num_steps_added=num_steps_added,
            num_tokens_added=num_tokens_added,
        )

        self.branches.append(branch)

        # Store for utility computation
        key = (prefix.trajectory_id, prefix.step_index, mode)
        self.branch_utilities[key] = utility

    def get_branches_for_prefix(
        self,
        prefix: Prefix,
    ) -> List[Branch]:
        """Get all branches for a specific prefix."""
        return [b for b in self.branches if b.prefix == prefix]

    def compute_branch_advantages(
        self,
        prefix: Prefix,
        epsilon: float = 1e-8,
    ) -> Dict[str, float]:
        """
        Compute relative advantages for branches from a prefix.

        Â_m = (U_m - mean(U)) / std(U)

        Args:
            prefix: The prefix
            epsilon: Small constant for numerical stability

        Returns:
            Dictionary mapping mode -> advantage
        """
        branches = self.get_branches_for_prefix(prefix)

        if not branches:
            return {}

        utilities = np.array([b.branch_utility for b in branches])
        mean_utility = np.mean(utilities)
        std_utility = np.std(utilities)

        advantages = {}
        for branch in branches:
            advantage = (branch.branch_utility - mean_utility) / (std_utility + epsilon)
            advantages[branch.forced_mode] = advantage

        return advantages

    def get_mode_target(self, prefix: Prefix) -> str:
        """
        Get the best mode for this prefix based on branch utilities.

        m_best = argmax_m U_m

        Args:
            prefix: The prefix

        Returns:
            Best mode ("THINK", "OBSERVE", or "ANSWER")
        """
        branches = self.get_branches_for_prefix(prefix)

        if not branches:
            return "ANSWER"

        best_branch = max(branches, key=lambda b: b.branch_utility)
        return best_branch.forced_mode

    def reset(self):
        """Clear all collected branches."""
        self.branches = []
        self.branch_utilities = {}

    def summary(self) -> Dict[str, Any]:
        """Get summary of collected branches."""
        if not self.branches:
            return {"num_branches": 0}

        modes = [b.forced_mode for b in self.branches]
        utilities = [b.branch_utility for b in self.branches]

        return {
            "num_branches": len(self.branches),
            "num_prefixes": len(set((b.prefix.trajectory_id, b.prefix.step_index) for b in self.branches)),
            "mode_distribution": {
                mode: sum(1 for m in modes if m == mode)
                for mode in set(modes)
            },
            "mean_utility": float(np.mean(utilities)),
            "std_utility": float(np.std(utilities)),
        }
