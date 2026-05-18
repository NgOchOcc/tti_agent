"""
Prefix mining and counterfactual branch generation for CVI-SDAR.

Implements:
1. Prefix selection: Choose informative prefixes from trajectories
2. Branch generation: Create forced continuations for each mode
3. Branch evaluation: Compute utilities for each branch
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import random


@dataclass
class Prefix:
    """Represents a trajectory prefix."""

    trajectory_id: str
    step_index: int
    history: Dict[str, Any]  # Partial history up to this prefix
    budget: Dict[str, float]  # Remaining budget
    metadata: Dict[str, Any] = None  # Additional metadata

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Branch:
    """Represents a counterfactual branch from a prefix."""

    prefix: Prefix
    forced_mode: int  # Mode that was forced for this branch
    full_trajectory: Dict[str, Any]  # Complete trajectory from prefix
    branch_utility: float = 0.0  # U_t^m
    num_steps_added: int = 0  # Steps taken from prefix
    num_tokens_added: int = 0  # Tokens generated from prefix


class PrefixMiner:
    """
    Mines informative prefixes from trajectories for CVI-SDAR training.

    Selects prefixes where the agent faces key decision points:
    - Where it naturally wants to stop
    - High-entropy mode/action choices
    - After page transitions
    - Before irreversible actions
    - In recovery situations
    """

    def __init__(
        self,
        sampling_strategy: str = "mixed",
        max_prefixes_per_trajectory: int = 5,
        stop_threshold: float = 0.7,  # Entropy threshold for stopping prefixes
        random_seed: int = 42,
    ):
        """
        Args:
            sampling_strategy: "random", "entropy", "position", or "mixed"
            max_prefixes_per_trajectory: Maximum prefixes to mine per trajectory
            stop_threshold: Threshold for detecting "stopping tendency" prefixes
            random_seed: Random seed for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.max_prefixes_per_trajectory = max_prefixes_per_trajectory
        self.stop_threshold = stop_threshold
        self.random_seed = random_seed
        random.seed(random_seed)

    def mine_prefixes(
        self,
        trajectory: Dict[str, Any],
        max_prefixes: Optional[int] = None,
    ) -> List[Prefix]:
        """
        Mine informative prefixes from a trajectory.

        Args:
            trajectory: Single trajectory dict with keys:
                - "task_id": Task identifier
                - "observations": List of observations
                - "actions": List of actions taken
                - "rewards": List of step rewards
                - "success": Final success boolean
                - "num_tokens": Tokens generated
                - "mode_distribution": Mode choices if available
                - etc.
            max_prefixes: Override max prefixes for this trajectory

        Returns:
            List of Prefix objects representing candidate decision points
        """
        if max_prefixes is None:
            max_prefixes = self.max_prefixes_per_trajectory

        prefixes = []
        trajectory_length = len(trajectory.get("actions", []))

        # Candidate indices based on strategy
        if self.sampling_strategy == "random":
            candidate_indices = self._get_random_indices(
                trajectory_length, max_prefixes
            )
        elif self.sampling_strategy == "entropy":
            candidate_indices = self._get_entropy_based_indices(
                trajectory, max_prefixes
            )
        elif self.sampling_strategy == "position":
            candidate_indices = self._get_position_based_indices(
                trajectory_length, max_prefixes
            )
        elif self.sampling_strategy == "mixed":
            candidate_indices = self._get_mixed_indices(
                trajectory, trajectory_length, max_prefixes
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        # Create Prefix objects
        for t in sorted(candidate_indices):
            if t >= trajectory_length:
                continue

            prefix = self._create_prefix(trajectory, t)
            if prefix is not None:
                prefixes.append(prefix)

        return prefixes[:max_prefixes]

    def _create_prefix(
        self,
        trajectory: Dict[str, Any],
        step_index: int
    ) -> Optional[Prefix]:
        """Create a Prefix object at a specific step."""
        trajectory_id = trajectory.get("task_id", "unknown")

        # Build partial history up to this step
        history = {
            "observations": trajectory.get("observations", [])[:step_index + 1],
            "actions": trajectory.get("actions", [])[:step_index],
            "modes": trajectory.get("modes", [])[:step_index] if "modes" in trajectory else [],
            "thoughts": trajectory.get("thoughts", [])[:step_index] if "thoughts" in trajectory else [],
        }

        # Estimate remaining budget (simplified)
        max_steps = trajectory.get("max_steps", 30)
        steps_taken = step_index
        remaining_budget = {
            "env_interactions": max(0, max_steps - steps_taken),
            "tokens": max(0, 4096 - trajectory.get("total_tokens_so_far", 0)),
            "verifier_calls": max(0, 10 - trajectory.get("verifier_calls_so_far", 0)),
            "risk_tolerance": 1.0,
        }

        metadata = {
            "trajectory_id": trajectory_id,
            "step_index": step_index,
            "success": trajectory.get("success", False),
            "total_steps": len(trajectory.get("actions", [])),
        }

        return Prefix(
            trajectory_id=trajectory_id,
            step_index=step_index,
            history=history,
            budget=remaining_budget,
            metadata=metadata
        )

    def _get_random_indices(
        self,
        trajectory_length: int,
        max_prefixes: int
    ) -> List[int]:
        """Randomly select prefix indices."""
        if trajectory_length <= max_prefixes:
            return list(range(trajectory_length))

        return sorted(random.sample(range(trajectory_length), max_prefixes))

    def _get_position_based_indices(
        self,
        trajectory_length: int,
        max_prefixes: int
    ) -> List[int]:
        """Select prefixes at regular positions throughout trajectory."""
        if trajectory_length <= max_prefixes:
            return list(range(trajectory_length))

        # Divide trajectory into segments and pick one prefix from each
        segment_size = trajectory_length // max_prefixes
        indices = [
            min(i * segment_size + segment_size // 2, trajectory_length - 1)
            for i in range(max_prefixes)
        ]

        return sorted(list(set(indices)))

    def _get_entropy_based_indices(
        self,
        trajectory: Dict[str, Any],
        max_prefixes: int
    ) -> List[int]:
        """
        Select prefixes at high-entropy (uncertain) decision points.

        Looks for steps where mode/action choice had high entropy.
        """
        actions = trajectory.get("actions", [])

        if not actions:
            return []

        # Extract entropy if available, otherwise use length as proxy
        entropies = trajectory.get("action_entropies", [])

        if not entropies:
            # Use action length as proxy for entropy (longer = more uncertain)
            entropies = [len(str(a)) for a in actions]

        # Select top-entropy indices
        entropy_indices = sorted(
            enumerate(entropies),
            key=lambda x: x[1],
            reverse=True
        )

        selected_indices = [idx for idx, _ in entropy_indices[:max_prefixes]]
        return sorted(selected_indices)

    def _get_mixed_indices(
        self,
        trajectory: Dict[str, Any],
        trajectory_length: int,
        max_prefixes: int
    ) -> List[int]:
        """
        Mix multiple strategies:
        1. High-entropy positions (50%)
        2. Random positions (25%)
        3. Regular positions (25%)
        """
        indices = set()

        # High-entropy
        entropy_indices = self._get_entropy_based_indices(trajectory, max_prefixes // 2)
        indices.update(entropy_indices)

        # Random
        random_indices = self._get_random_indices(
            trajectory_length, max_prefixes // 4
        )
        indices.update(random_indices)

        # Regular positions
        position_indices = self._get_position_based_indices(
            trajectory_length, max_prefixes // 4
        )
        indices.update(position_indices)

        # Special prefixes: after page transitions, before forms, etc
        special_indices = self._get_special_indices(trajectory, max_prefixes // 4)
        indices.update(special_indices)

        return sorted(list(indices))[:max_prefixes]

    def _get_special_indices(
        self,
        trajectory: Dict[str, Any],
        max_prefixes: int
    ) -> List[int]:
        """
        Find special prefixes:
        - After page transitions
        - Before form submissions
        - In failure regions
        - Before recovery attempts
        """
        special_indices = []

        observations = trajectory.get("observations", [])
        actions = trajectory.get("actions", [])
        modes = trajectory.get("modes", [])

        for t in range(len(actions)):
            is_special = False

            # Check for page transitions
            if t > 0:
                curr_url = observations[t].get("url", "") if isinstance(observations[t], dict) else ""
                prev_url = observations[t-1].get("url", "") if isinstance(observations[t-1], dict) else ""

                if curr_url != prev_url:
                    is_special = True

            # Check for form submissions or irreversible actions
            action_str = str(actions[t]).lower()
            if any(keyword in action_str for keyword in ["submit", "send", "delete", "confirm"]):
                is_special = True

            # Check for recovery attempts
            if modes and t < len(modes) and modes[t] == 4:  # RECOVER mode
                is_special = True

            if is_special:
                special_indices.append(t)

        return special_indices[:max_prefixes]


class BranchGenerator:
    """
    Generates counterfactual branches from prefixes.

    For each prefix and each mode, creates a forced continuation
    where the agent is forced to take that mode first.
    """

    def __init__(
        self,
        num_modes: int = 6,
        max_branch_length: Optional[int] = None,
        use_privileged_budget: bool = True,
    ):
        """
        Args:
            num_modes: Number of modes (default 6)
            max_branch_length: Maximum length for each branch (None = use trajectory length)
            use_privileged_budget: If True, give teacher access to extra budget
        """
        self.num_modes = num_modes
        self.max_branch_length = max_branch_length
        self.use_privileged_budget = use_privileged_budget

    def generate_branches(
        self,
        prefix: Prefix,
        full_trajectory: Dict[str, Any],
        sampled_modes: Optional[List[int]] = None,
    ) -> List[Branch]:
        """
        Generate forced branches for each mode.

        Args:
            prefix: Prefix object
            full_trajectory: Full trajectory that this prefix comes from
            sampled_modes: Specific modes to sample (None = all modes)

        Returns:
            List of Branch objects
        """
        if sampled_modes is None:
            sampled_modes = list(range(self.num_modes))

        branches = []

        for mode in sampled_modes:
            # For now, we use the continuation from the full trajectory
            # In practice, you might force the mode and re-run the agent
            branch = self._create_forced_branch(
                prefix, full_trajectory, mode
            )

            if branch is not None:
                branches.append(branch)

        return branches

    def _create_forced_branch(
        self,
        prefix: Prefix,
        full_trajectory: Dict[str, Any],
        forced_mode: int,
    ) -> Optional[Branch]:
        """Create a branch where a specific mode is forced."""
        # Use the continuation from the full trajectory starting at this prefix
        start_idx = prefix.step_index

        # Simulate the trajectory from this prefix onward
        # In practice, this would require:
        # 1. Restore environment state at this prefix
        # 2. Force the mode to be selected
        # 3. Re-run the policy
        # 4. Collect the new trajectory

        # For now, we use the original trajectory continuation
        continuation = {
            "actions": full_trajectory.get("actions", [])[start_idx:],
            "observations": full_trajectory.get("observations", [])[start_idx + 1:],
            "rewards": full_trajectory.get("rewards", [])[start_idx:],
            "num_tokens": full_trajectory.get("num_tokens", 0) - full_trajectory.get("tokens_before_prefix", 0),
        }

        # Calculate branch utility
        # U_t^m = R(τ_t^m) - Cost(τ_t^m)
        success = full_trajectory.get("success", 0)  # Terminal reward

        # Cost calculation
        num_env_steps = len(continuation["actions"])
        num_tokens = continuation["num_tokens"]
        # These would normally come from the environment execution

        branch_utility = success - (0.01 * num_env_steps + 0.001 * num_tokens)

        return Branch(
            prefix=prefix,
            forced_mode=forced_mode,
            full_trajectory=full_trajectory,
            branch_utility=branch_utility,
            num_steps_added=num_env_steps,
            num_tokens_added=num_tokens,
        )

    def compute_teacher_mode(self, branches: List[Branch]) -> Tuple[int, float]:
        """
        Compute the teacher mode (best mode) for a prefix.

        m_T* = argmax_m U_t^m

        Args:
            branches: List of branches for different modes

        Returns:
            (teacher_mode_index, max_utility)
        """
        if not branches:
            return 0, 0.0

        utilities = [b.branch_utility for b in branches]
        best_idx = np.argmax(utilities)

        return best_idx, utilities[best_idx]

    def compute_cvi_advantage(
        self,
        branches: List[Branch],
        teacher_mode_idx: int,
    ) -> float:
        """
        Compute empirical counterfactual advantage of teacher mode over ANSWER.

        Â_t^CVI(m_T*) = U_t^m_T* - U_t^ANSWER

        Args:
            branches: List of branches
            teacher_mode_idx: Index of teacher mode

        Returns:
            CVI advantage value
        """
        # Find ANSWER branch (last mode by convention)
        answer_utility = None
        teacher_utility = None

        for branch in branches:
            if branch.forced_mode == self.num_modes - 1:  # ANSWER mode
                answer_utility = branch.branch_utility
            if branch.forced_mode == teacher_mode_idx:
                teacher_utility = branch.branch_utility

        if answer_utility is None or teacher_utility is None:
            return 0.0

        return teacher_utility - answer_utility
