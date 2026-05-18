"""
Inference algorithm for TOS-RL.

At each step:
1. Condition LLM on history and remaining budget
2. LLM emits one of three mode tokens: [THINK], [OBSERVE], [ANSWER]
3. Execute mode and update state
"""

from enum import Enum
from typing import Dict, Optional, Any, Tuple
import logging

from .utils import BudgetState, TokenModeFormatter

logger = logging.getLogger(__name__)


class TokenMode(str, Enum):
    """The three mode tokens in TOS-RL."""

    THINK = "THINK"
    OBSERVE = "OBSERVE"
    ANSWER = "ANSWER"


class TOSRLInference:
    """
    Inference controller for TOS-RL.

    Simple algorithm:
    1. Include budget in prompt
    2. LLM generates mode token
    3. Execute mode
    4. Update state
    """

    def __init__(
        self,
        max_steps: int = 30,
        max_tokens: int = 4096,
        max_time: float = 300.0,
    ):
        """
        Initialize inference controller.

        Args:
            max_steps: Maximum environment interactions
            max_tokens: Maximum tokens to generate
            max_time: Maximum wall-clock time in seconds
        """
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_time = max_time

    def create_prompt_with_budget(
        self,
        task: str,
        history: str,
        budget: BudgetState,
    ) -> str:
        """
        Create full prompt including budget constraints.

        Args:
            task: Original task instruction
            history: Conversation history so far
            budget: Current budget state

        Returns:
            Full prompt for LLM
        """
        budget_suffix = budget.format_for_prompt()

        prompt = f"""Task: {task}

{history}

{budget_suffix}

At each step, emit one of: [THINK], [OBSERVE], or [ANSWER]
[THINK]: reason internally using tokens
[OBSERVE]: interact with browser for new information
[ANSWER]: stop and give final answer

Next action:"""

        return prompt

    def parse_llm_output(
        self,
        output: str,
    ) -> Tuple[Optional[str], str]:
        """
        Parse LLM output to extract mode and content.

        Args:
            output: Raw LLM output

        Returns:
            (mode, content) where mode is "THINK", "OBSERVE", or "ANSWER"
        """
        mode, content = TokenModeFormatter.parse_mode_and_content(output)
        return mode, content

    def should_continue(
        self,
        mode: Optional[str],
        budget: BudgetState,
    ) -> bool:
        """
        Determine whether to continue or stop.

        Args:
            mode: The selected mode ("THINK", "OBSERVE", or "ANSWER")
            budget: Current budget state

        Returns:
            True if should continue, False if should stop
        """
        # Explicit ANSWER mode always stops
        if mode == TokenMode.ANSWER.value:
            return False

        # Check budget exhaustion
        if budget.is_exhausted():
            logger.warning("Budget exhausted, forcing ANSWER")
            return False

        # Continue for THINK and OBSERVE
        return True

    def update_budget(
        self,
        budget: BudgetState,
        mode: str,
        num_tokens_generated: int,
    ) -> BudgetState:
        """
        Update budget after executing a mode.

        Args:
            budget: Current budget
            mode: Mode that was executed
            num_tokens_generated: Tokens generated in this step

        Returns:
            Updated budget
        """
        new_budget = budget.copy()

        # All modes consume tokens
        new_budget.tokens -= num_tokens_generated

        # OBSERVE consumes environment step
        if mode == TokenMode.OBSERVE.value:
            new_budget.env_interactions -= 1

        # Rough time estimate
        new_budget.time -= num_tokens_generated / 100  # ~100 tokens/sec

        return new_budget

    def run_episode(
        self,
        task: str,
        llm_fn,  # Function that takes prompt and returns output
        initial_history: str = "",
        initial_budget: Optional[BudgetState] = None,
        max_episode_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a single episode of inference.

        Args:
            task: Task instruction
            llm_fn: Function(prompt: str) -> output: str
            initial_history: Initial conversation history
            initial_budget: Initial budget (default: full)
            max_episode_steps: Max steps (default: self.max_steps)

        Returns:
            Episode result dict with:
                - "success": Whether task succeeded
                - "history": Full conversation history
                - "modes": Sequence of modes executed
                - "num_steps": Total environment steps
                - "num_tokens": Total tokens generated
                - "final_answer": Final answer given
        """
        if initial_budget is None:
            initial_budget = BudgetState(
                env_interactions=self.max_steps,
                tokens=self.max_tokens,
                time=self.max_time,
            )

        if max_episode_steps is None:
            max_episode_steps = self.max_steps

        budget = initial_budget.copy()
        history = initial_history
        modes = []
        total_tokens = 0
        num_env_steps = 0

        for step in range(max_episode_steps):
            # Create prompt with budget
            prompt = self.create_prompt_with_budget(task, history, budget)

            # Get LLM output
            try:
                llm_output = llm_fn(prompt)
            except Exception as e:
                logger.error(f"LLM error at step {step}: {e}")
                break

            # Parse mode and content
            mode, content = self.parse_llm_output(llm_output)

            if mode is None:
                logger.warning(f"Failed to parse mode from: {llm_output[:100]}")
                mode = TokenMode.ANSWER.value
                content = llm_output

            modes.append(mode)

            # Count tokens
            num_tokens = len(llm_output.split())  # Approximate
            total_tokens += num_tokens

            # Update history
            history += f"\nAssistant: {llm_output}"

            # Update budget
            budget = self.update_budget(budget, mode, num_tokens)

            # Count environment steps
            if mode == TokenMode.OBSERVE.value:
                num_env_steps += 1

            # Check if should continue
            if not self.should_continue(mode, budget):
                logger.info(f"Episode ended at step {step} with mode {mode}")
                break

            # In practice, would get environment observation here
            # For now, just append placeholder
            history += "\n[Environment: <observation would go here>]"

        return {
            "success": False,  # Would compute from task evaluation
            "history": history,
            "modes": modes,
            "num_steps": num_env_steps,
            "num_tokens": total_tokens,
            "final_answer": content,
            "final_budget": budget.as_dict(),
        }


class InferenceStatistics:
    """Tracks inference statistics across episodes."""

    def __init__(self):
        """Initialize tracker."""
        self.episodes = []
        self.mode_counts = {
            TokenMode.THINK.value: 0,
            TokenMode.OBSERVE.value: 0,
            TokenMode.ANSWER.value: 0,
        }

    def record_episode(self, result: Dict[str, Any]):
        """Record an episode result."""
        self.episodes.append(result)

        # Count modes
        for mode in result.get("modes", []):
            if mode in self.mode_counts:
                self.mode_counts[mode] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.episodes:
            return {}

        num_episodes = len(self.episodes)
        num_steps = [e.get("num_steps", 0) for e in self.episodes]
        num_tokens = [e.get("num_tokens", 0) for e in self.episodes]

        total_modes = sum(self.mode_counts.values())

        return {
            "num_episodes": num_episodes,
            "mean_steps": sum(num_steps) / num_episodes if num_episodes else 0,
            "mean_tokens": sum(num_tokens) / num_episodes if num_episodes else 0,
            "mode_distribution": {
                mode: count / total_modes if total_modes > 0 else 0
                for mode, count in self.mode_counts.items()
            },
            "total_modes_used": total_modes,
        }

    def reset(self):
        """Reset tracker."""
        self.episodes = []
        self.mode_counts = {
            TokenMode.THINK.value: 0,
            TokenMode.OBSERVE.value: 0,
            TokenMode.ANSWER.value: 0,
        }
