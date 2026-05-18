"""
TOS-RL: LLM-Only Think-Observe-Stop Reinforcement Learning

A minimal framework for training a single language model to decide when to:
- THINK: Internal reasoning without environment interaction
- OBSERVE: Interact with the environment for new information
- ANSWER: Stop and submit the final answer

Unlike modular approaches, TOS-RL has NO external gate, critic, or router.
Only the LLM parameters are trained. Mode tokens are part of the LLM output.

Core modules:
- objectives: Cost-aware utility and loss functions
- optimization: Group-relative policy optimization (GRPO)
- branching: Prefix-level counterfactual branching
- training: Main training loop
- inference: Inference algorithm
- utils: Utilities and helpers
"""

from .objectives import (
    CostAwareUtility,
    compute_cost_aware_utility,
    batch_compute_utilities,
    normalize_utilities,
)
from .optimization import (
    GroupRelativePolicyOptimization,
    GRPOLoss,
    GRPOConfig,
)
from .branching import (
    PrefixBrancher,
    BranchCollector,
    Prefix,
    Branch,
)
from .training import TOSRLTrainer, TOSRLConfig
from .inference import TOSRLInference, TokenMode
from .utils import (
    BudgetState,
    CostMetrics,
    TokenModeFormatter,
    CostCoefficientSchedule,
    create_budget_prompt_suffix,
)

__version__ = "1.0.0"

__all__ = [
    # Objectives
    "CostAwareUtility",
    "compute_cost_aware_utility",
    "batch_compute_utilities",
    "normalize_utilities",

    # Optimization
    "GroupRelativePolicyOptimization",
    "GRPOLoss",
    "GRPOConfig",

    # Branching
    "PrefixBrancher",
    "BranchCollector",
    "Prefix",
    "Branch",

    # Training
    "TOSRLTrainer",
    "TOSRLConfig",

    # Inference
    "TOSRLInference",
    "TokenMode",

    # Utils
    "BudgetState",
    "CostMetrics",
    "TokenModeFormatter",
    "CostCoefficientSchedule",
    "create_budget_prompt_suffix",
]
