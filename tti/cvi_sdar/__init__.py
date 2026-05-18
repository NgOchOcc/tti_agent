"""
CVI-SDAR: Counterfactual Value-of-Interaction Self-Distilled Agentic RL

A reinforcement learning framework for training agents to decide when to think,
when to observe, and when to stop, based on learning counterfactual value-of-interaction.

Core modules:
- critics: Value functions (V, Q, V_ans)
- mode_policy: High-level mode selection policy
- distillation: CVI-gated self-distillation losses
- prefix_mining: Prefix selection and branch generation
- training: Main training loop
- inference: Inference algorithm
- utils: Helper functions

Example usage:
    from tti.cvi_sdar import CVISARTrainer, CVISARConfig

    config = CVISARConfig(
        state_embedding_dim=768,
        num_modes=6,
        device=torch.device("cuda:0")
    )

    trainer = CVISARTrainer(config)

    # Training loop
    for epoch in range(num_epochs):
        batch = prepare_batch(trajectories)
        prefix_data = generate_prefix_branches(trajectories)
        losses = trainer.train_step(batch, prefix_data)
"""

# Core imports
from .critics import CriticNetwork, MultiHeadCriticNetwork, create_critics, CriticLosses
from .mode_policy import Mode, ModePolicy, ModeActionPolicy, PPOModePolicy, MODE_NAMES, NUM_MODES
from .distillation import CVIDistillationLoss, CVIGate
from .prefix_mining import PrefixMiner, BranchGenerator, Prefix, Branch
from .training import CVISARTrainer, CVISARConfig
from .inference import CVISARInferenceController, InferenceState, InferenceDecision, DeltaSchedule, InferenceTracker
from .utils import Budget, CostMetrics, compute_cvi_gap, select_mode_by_cvi

__version__ = "0.1.0"

__all__ = [
    # Critics
    "CriticNetwork",
    "MultiHeadCriticNetwork",
    "create_critics",
    "CriticLosses",

    # Mode Policy
    "Mode",
    "ModePolicy",
    "ModeActionPolicy",
    "PPOModePolicy",
    "MODE_NAMES",
    "NUM_MODES",

    # Distillation
    "CVIDistillationLoss",
    "CVIGate",

    # Prefix Mining
    "PrefixMiner",
    "BranchGenerator",
    "Prefix",
    "Branch",

    # Training
    "CVISARTrainer",
    "CVISARConfig",

    # Inference
    "CVISARInferenceController",
    "InferenceState",
    "InferenceDecision",
    "DeltaSchedule",
    "InferenceTracker",

    # Utils
    "Budget",
    "CostMetrics",
    "compute_cvi_gap",
    "select_mode_by_cvi",
]
