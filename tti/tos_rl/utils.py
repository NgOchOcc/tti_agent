"""
Utilities for TOS-RL training and inference.

Includes:
- Budget state management
- Cost metrics tracking
- Mode token parsing/formatting
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import copy


@dataclass
class BudgetState:
    """State of remaining budget during interaction."""

    env_interactions: float = 30  # Remaining browser steps
    tokens: float = 4096  # Remaining tokens
    time: float = 300.0  # Remaining time in seconds

    def copy(self) -> "BudgetState":
        """Create a copy of this budget."""
        return copy.deepcopy(self)

    def as_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "env_interactions": self.env_interactions,
            "tokens": self.tokens,
            "time": self.time,
        }

    def format_for_prompt(self) -> str:
        """Format budget for inclusion in LLM prompt."""
        return (
            f"Remaining browser steps: {int(self.env_interactions)}. "
            f"Remaining tokens: {int(self.tokens)}. "
            f"Remaining time: {self.time:.0f}s."
        )

    def is_exhausted(self) -> bool:
        """Check if any budget is exhausted."""
        return (
            self.env_interactions <= 0 or
            self.tokens <= 0 or
            self.time <= 0
        )

    def __repr__(self) -> str:
        return (
            f"Budget(env={self.env_interactions:.0f}, "
            f"tok={self.tokens:.0f}, time={self.time:.1f}s)"
        )


@dataclass
class CostMetrics:
    """Metrics for cost tracking."""

    num_env_steps: int = 0
    num_tokens_generated: int = 0
    num_loops: int = 0
    num_invalid_actions: int = 0
    num_thinks: int = 0
    num_observes: int = 0
    num_answers: int = 0

    def total_cost(
        self,
        lambda_env: float = 0.01,
        lambda_tok: float = 0.001,
        lambda_loop: float = 0.1,
        lambda_bad: float = 0.2,
    ) -> float:
        """Compute total cost."""
        return (
            lambda_env * self.num_env_steps +
            lambda_tok * self.num_tokens_generated +
            lambda_loop * self.num_loops +
            lambda_bad * self.num_invalid_actions
        )

    def __repr__(self) -> str:
        return (
            f"Costs(steps={self.num_env_steps}, "
            f"tokens={self.num_tokens_generated}, "
            f"loops={self.num_loops}, invalid={self.num_invalid_actions})"
        )


class TokenModeFormatter:
    """
    Utilities for mode token formatting in LLM output.

    Mode tokens in TOS-RL:
    - [THINK]: Internal reasoning scratchpad
    - [OBSERVE]: Browser action request
    - [ANSWER]: Final answer/submission
    """

    THINK = "[THINK]"
    OBSERVE = "[OBSERVE]"
    ANSWER = "[ANSWER]"

    TOKEN_TO_MODE = {
        THINK: "THINK",
        OBSERVE: "OBSERVE",
        ANSWER: "ANSWER",
    }

    MODE_TO_TOKEN = {
        "THINK": THINK,
        "OBSERVE": OBSERVE,
        "ANSWER": ANSWER,
    }

    @classmethod
    def is_mode_token(cls, token: str) -> bool:
        """Check if a token is a mode token."""
        return token in cls.TOKEN_TO_MODE

    @classmethod
    def extract_mode(cls, text: str) -> Optional[str]:
        """
        Extract mode from text output.

        Returns first mode token found, or None.
        """
        for mode_token, mode in cls.TOKEN_TO_MODE.items():
            if text.startswith(mode_token):
                return mode

        return None

    @classmethod
    def format_with_mode(cls, mode: str, content: str) -> str:
        """Format content with mode prefix."""
        token = cls.MODE_TO_TOKEN.get(mode, cls.ANSWER)
        return f"{token} {content}"

    @classmethod
    def parse_mode_and_content(cls, text: str) -> Tuple[Optional[str], str]:
        """
        Parse mode and content from output.

        Returns:
            (mode, remaining_content)
        """
        mode = cls.extract_mode(text)

        if mode is None:
            return None, text

        # Remove mode token
        mode_token = cls.MODE_TO_TOKEN[mode]
        content = text[len(mode_token):].strip()

        return mode, content


class CostCoefficientSchedule:
    """
    Generates cost coefficients according to a schedule.

    During training, randomize cost coefficients to train a single
    policy that can adapt to different cost-efficiency tradeoffs.
    """

    def __init__(
        self,
        lambda_env_range: Tuple[float, float] = (0.001, 0.1),
        lambda_tok_range: Tuple[float, float] = (0.0001, 0.01),
        lambda_loop_range: Tuple[float, float] = (0.01, 0.5),
        lambda_bad_range: Tuple[float, float] = (0.1, 1.0),
    ):
        """Initialize schedule with ranges."""
        self.lambda_env_range = lambda_env_range
        self.lambda_tok_range = lambda_tok_range
        self.lambda_loop_range = lambda_loop_range
        self.lambda_bad_range = lambda_bad_range

    def sample(self) -> Dict[str, float]:
        """Sample cost coefficients from ranges."""
        import random

        return {
            "env": random.uniform(*self.lambda_env_range),
            "tok": random.uniform(*self.lambda_tok_range),
            "loop": random.uniform(*self.lambda_loop_range),
            "bad": random.uniform(*self.lambda_bad_range),
        }

    def get_defaults(self) -> Dict[str, float]:
        """Get default (middle) cost coefficients."""
        return {
            "env": (self.lambda_env_range[0] + self.lambda_env_range[1]) / 2,
            "tok": (self.lambda_tok_range[0] + self.lambda_tok_range[1]) / 2,
            "loop": (self.lambda_loop_range[0] + self.lambda_loop_range[1]) / 2,
            "bad": (self.lambda_bad_range[0] + self.lambda_bad_range[1]) / 2,
        }


def create_budget_prompt_suffix(budget: BudgetState) -> str:
    """
    Create a prompt suffix instructing the LLM about budget constraints.

    This is included in the context to make the LLM budget-aware.
    """
    return (
        "\nYou have the following budget:\n"
        + budget.format_for_prompt()
        + "\n"
        "At each step, you must choose one of three actions:\n"
        "[THINK]: Use tokens to reason internally without browser interaction.\n"
        "[OBSERVE]: Interact with the browser (click, scroll, search, etc).\n"
        "[ANSWER]: Stop and submit your final answer.\n"
        "Choose wisely to optimize for task success while respecting the budget."
    )
