"""
Inference algorithm for CVI-SDAR.

Implements the inference-time decision process:
1. Estimate V_ans and Q for each mode at the current state
2. Compute CVI gaps: Δ_m = Q(h,b,m) - V_ans(h,b)
3. Select best mode if max Δ_m > δ(b), otherwise answer
4. Execute the mode and collect new observation
"""

import torch
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from .mode_policy import Mode, MODE_NAMES
from .utils import Budget


@dataclass
class InferenceState:
    """State during CVI-SDAR inference."""

    history: Dict[str, Any]  # Current trajectory history
    budget: Budget  # Remaining budget
    step_count: int = 0
    mode_counts: Dict[int, int] = None  # Count of each mode used

    def __post_init__(self):
        if self.mode_counts is None:
            self.mode_counts = {i: 0 for i in range(6)}


@dataclass
class InferenceDecision:
    """Decision made during inference."""

    mode: int  # Selected mode (0-5)
    cvi_gaps: np.ndarray  # [num_modes] CVI gaps
    v_ans_value: float  # V_ans at this state
    q_values: np.ndarray  # [num_modes] Q values
    should_continue: bool  # Whether to continue or answer
    confidence: float  # How confident is the decision (max gap / second max gap)


class CVISARInferenceController:
    """
    Inference controller for CVI-SDAR.

    At each step, computes CVI gaps and decides which mode to take.
    """

    def __init__(
        self,
        mode_policy: torch.nn.Module,
        critics: Dict[str, torch.nn.Module],
        delta_threshold: float = 0.0,
        delta_schedule: Optional[callable] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            mode_policy: Trained mode policy π_φ
            critics: Dictionary with "v", "q", "v_ans" critics
            delta_threshold: Constant threshold δ(b) for continuing
            delta_schedule: Optional function δ(budget) -> threshold
            device: Device for inference
        """
        self.mode_policy = mode_policy
        self.critics = critics
        self.delta_threshold = delta_threshold
        self.delta_schedule = delta_schedule
        self.device = device

        # Set to eval mode
        self.mode_policy.eval()
        for critic in self.critics.values():
            critic.eval()

    def get_cvi_decision(
        self,
        state_embedding: torch.Tensor,  # [1, embedding_dim] or [embedding_dim]
        budget: Optional[Budget] = None,
        deterministic: bool = False,
    ) -> InferenceDecision:
        """
        Get CVI-based decision at current state.

        Args:
            state_embedding: Encoded representation of current state
            budget: Current budget state (for dynamic threshold)
            deterministic: If True, take argmax for mode

        Returns:
            InferenceDecision with mode, gaps, and other info
        """
        # Ensure state is batched
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)

        with torch.no_grad():
            # Get critic estimates
            v_ans = self.critics["v_ans"](state_embedding)  # [1]
            q_values = self.critics["q"](state_embedding)  # [1, num_modes]

            v_ans_value = v_ans.item()
            q_values_np = q_values.cpu().numpy()[0]  # [num_modes]

        # Compute CVI gaps: Δ_m = Q_m - V_ans
        cvi_gaps = q_values_np - v_ans_value  # [num_modes]

        # Determine threshold
        if self.delta_schedule is not None and budget is not None:
            threshold = self.delta_schedule(budget)
        else:
            threshold = self.delta_threshold

        # Select best mode
        best_mode = np.argmax(cvi_gaps)
        best_gap = cvi_gaps[best_mode]

        # Decide whether to continue
        should_continue = best_gap > threshold

        # If should answer, set mode to ANSWER
        if not should_continue:
            best_mode = 5  # ANSWER mode (last mode by convention)

        # Compute confidence
        sorted_gaps = np.sort(cvi_gaps)
        if len(sorted_gaps) >= 2:
            confidence = sorted_gaps[-1] - sorted_gaps[-2]
        else:
            confidence = sorted_gaps[-1]

        return InferenceDecision(
            mode=best_mode,
            cvi_gaps=cvi_gaps,
            v_ans_value=v_ans_value,
            q_values=q_values_np,
            should_continue=should_continue,
            confidence=float(confidence),
        )

    def execute_step(
        self,
        state: InferenceState,
        decision: InferenceDecision,
        action_policy: Optional[torch.nn.Module] = None,
        state_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[InferenceState, Dict[str, Any]]:
        """
        Execute one step of inference.

        Args:
            state: Current inference state
            decision: CVI decision for this step
            action_policy: Policy for generating low-level actions (optional)
            state_embedding: Encoded state (for action generation)

        Returns:
            (updated_state, step_info)
        """
        mode = decision.mode
        mode_name = MODE_NAMES.get(mode, "UNKNOWN")

        # Update state
        state.step_count += 1
        state.mode_counts[mode] += 1

        step_info = {
            "step": state.step_count,
            "mode": mode,
            "mode_name": mode_name,
            "cvi_gaps": decision.cvi_gaps,
            "v_ans": decision.v_ans_value,
            "q_values": decision.q_values,
            "should_continue": decision.should_continue,
            "confidence": decision.confidence,
            "budget": state.budget.as_dict(),
        }

        # In actual implementation, would:
        # 1. Execute the mode in the environment
        # 2. Collect new observation
        # 3. Update state.history
        # 4. Update state.budget

        return state, step_info


class DeltaSchedule:
    """
    Budget-dependent thresholds δ(b) for stopping decisions.

    Different strategies for deciding when to stop based on remaining budget.
    """

    @staticmethod
    def constant(threshold: float = 0.0) -> callable:
        """Constant threshold regardless of budget."""
        def schedule(budget: Budget) -> float:
            return threshold
        return schedule

    @staticmethod
    def budget_linear(
        initial: float = 0.1,
        max_threshold: float = 1.0,
    ) -> callable:
        """Linear increase in threshold as budget depletes."""
        def schedule(budget: Budget) -> float:
            # Use env_interactions as primary budget
            pct_remaining = budget.env_interactions / 100.0  # Assuming max is 100
            pct_remaining = np.clip(pct_remaining, 0, 1)

            threshold = initial + (1 - pct_remaining) * (max_threshold - initial)
            return threshold

        return schedule

    @staticmethod
    def aggressive_stopping(
        low_budget_threshold: float = 1.0,
        budget_threshold: int = 5,
    ) -> callable:
        """
        Aggressive stopping when budget is very low.

        If remaining budget < threshold, force answering.
        """
        def schedule(budget: Budget) -> float:
            if budget.env_interactions < budget_threshold:
                return low_budget_threshold

            return 0.0

        return schedule

    @staticmethod
    def entropy_adaptive(
        mode_entropy: float,
        entropy_threshold: float = 0.5,
        high_entropy_threshold: float = 0.5,
        low_entropy_threshold: float = -0.5,
    ) -> callable:
        """
        Threshold based on mode policy entropy.

        High entropy (uncertain) -> higher threshold (more conservative)
        Low entropy (certain) -> lower threshold (more aggressive)
        """
        def schedule(budget: Budget) -> float:
            if mode_entropy > entropy_threshold:
                return high_entropy_threshold
            else:
                return low_entropy_threshold

        return schedule


class InferenceTracker:
    """
    Tracks inference statistics for monitoring and analysis.
    """

    def __init__(self):
        """Initialize tracker."""
        self.steps = []
        self.mode_counts = {i: 0 for i in range(6)}
        self.total_steps = 0
        self.cvi_gaps_history = []
        self.v_ans_history = []
        self.decisions_history = []

    def record_step(self, info: Dict[str, Any]):
        """Record a step's information."""
        self.steps.append(info)
        self.mode_counts[info["mode"]] += 1
        self.total_steps += 1
        self.cvi_gaps_history.append(info["cvi_gaps"])
        self.v_ans_history.append(info["v_ans"])

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        mode_percentages = {
            mode: (count / self.total_steps * 100) if self.total_steps > 0 else 0
            for mode, count in self.mode_counts.items()
        }

        cvi_gaps_mean = np.mean([g.max() for g in self.cvi_gaps_history]) if self.cvi_gaps_history else 0
        v_ans_mean = np.mean(self.v_ans_history) if self.v_ans_history else 0

        return {
            "total_steps": self.total_steps,
            "mode_counts": self.mode_counts,
            "mode_percentages": mode_percentages,
            "avg_max_cvi_gap": float(cvi_gaps_mean),
            "avg_v_ans": float(v_ans_mean),
        }

    def get_mode_names(self) -> Dict[str, int]:
        """Get mode name to count mapping."""
        return {
            MODE_NAMES.get(mode, f"Mode{mode}"): count
            for mode, count in self.mode_counts.items()
        }
