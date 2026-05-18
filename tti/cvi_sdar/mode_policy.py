"""
Mode policy for CVI-SDAR.

Implements the high-level policy π_φ(m_t | h_t, b_t) that selects modes:
- THINK: Internal reasoning without environment interaction
- OBSERVE: Acquire extra information from environment
- ACT: Task-progressing action
- VERIFY: Verification/cross-checking
- RECOVER: Backtrack/recovery
- ANSWER: Stop and submit answer
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
from enum import IntEnum


class Mode(IntEnum):
    """Enumeration of test-time operation modes."""
    THINK = 0
    OBSERVE = 1
    ACT = 2
    VERIFY = 3
    RECOVER = 4
    ANSWER = 5


MODE_NAMES = {
    Mode.THINK: "THINK",
    Mode.OBSERVE: "OBSERVE",
    Mode.ACT: "ACT",
    Mode.VERIFY: "VERIFY",
    Mode.RECOVER: "RECOVER",
    Mode.ANSWER: "ANSWER",
}

NUM_MODES = 6


class ModePolicy(nn.Module):
    """
    High-level mode selection policy.

    Maps state embeddings to mode probability distributions.
    Can be used as:
    1. A separate head on top of shared encoding
    2. A standalone small network
    3. Part of a larger policy network
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128,),
        dropout: float = 0.1,
        num_modes: int = NUM_MODES,
        temperature: float = 1.0,
    ):
        """
        Args:
            input_dim: Dimension of input (state embedding)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            num_modes: Number of modes (default 6)
            temperature: Temperature for softmax (for exploration control)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_modes = num_modes
        self.temperature = temperature

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output logits for each mode
        layers.append(nn.Linear(prev_dim, num_modes))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state_embedding: torch.Tensor,
        return_logits: bool = False,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mode probabilities.

        Args:
            state_embedding: [batch_size, input_dim] state encoding
            return_logits: If True, also return logits
            temperature: Override temperature for this forward pass

        Returns:
            probs: [batch_size, num_modes] probability distribution over modes
            logits: [batch_size, num_modes] raw logits (if return_logits=True)
        """
        logits = self.network(state_embedding)  # [batch_size, num_modes]

        if temperature is None:
            temperature = self.temperature

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)  # [batch_size, num_modes]

        if return_logits:
            return probs, logits
        return probs

    def sample_mode(
        self,
        state_embedding: torch.Tensor,
        temperature: Optional[float] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a mode from the policy.

        Args:
            state_embedding: [batch_size, input_dim]
            temperature: Temperature override
            deterministic: If True, take argmax instead of sampling

        Returns:
            modes: [batch_size] sampled mode indices
            log_probs: [batch_size] log probabilities of sampled modes
        """
        probs = self.forward(state_embedding, temperature=temperature)

        if deterministic:
            modes = torch.argmax(probs, dim=-1)
        else:
            modes = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Get log probabilities of sampled modes
        log_probs = torch.log(probs[torch.arange(len(modes)), modes] + 1e-8)

        return modes, log_probs

    def get_mode_logprob(
        self,
        state_embedding: torch.Tensor,
        modes: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get log probability of specific modes.

        Args:
            state_embedding: [batch_size, input_dim]
            modes: [batch_size] mode indices
            temperature: Temperature override

        Returns:
            log_probs: [batch_size] log probabilities
        """
        probs = self.forward(state_embedding, temperature=temperature)
        log_probs = torch.log(probs[torch.arange(len(modes)), modes] + 1e-8)
        return log_probs


class ModeActionPolicy(nn.Module):
    """
    Combined mode + action policy.

    First samples a mode, then samples an action conditioned on the mode.
    This follows the decomposition in the proposal:
    - π_φ(m_t | h_t, b_t): mode policy
    - π_θ(a_t | h_t, b_t, m_t): action policy conditioned on mode
    """

    def __init__(
        self,
        state_dim: int,
        mode_hidden_dims: Tuple[int, ...] = (128,),
        num_modes: int = NUM_MODES,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        """
        Args:
            state_dim: Dimension of state embedding
            mode_hidden_dims: Hidden dimensions for mode policy
            num_modes: Number of modes
            dropout: Dropout rate
            temperature: Softmax temperature
        """
        super().__init__()
        self.num_modes = num_modes

        # Mode policy
        self.mode_policy = ModePolicy(
            input_dim=state_dim,
            hidden_dims=mode_hidden_dims,
            dropout=dropout,
            num_modes=num_modes,
            temperature=temperature
        )

    def forward(
        self,
        state_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get mode probabilities.

        Args:
            state_embedding: [batch_size, state_dim]

        Returns:
            Dictionary with:
                - "mode_probs": [batch_size, num_modes]
                - "mode_logits": [batch_size, num_modes]
        """
        probs, logits = self.mode_policy.forward(
            state_embedding,
            return_logits=True
        )

        return {
            "mode_probs": probs,
            "mode_logits": logits,
        }

    def sample(
        self,
        state_embedding: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Sample mode and action.

        Args:
            state_embedding: [batch_size, state_dim]
            deterministic: If True, take argmax for mode

        Returns:
            Dictionary with:
                - "modes": [batch_size] sampled modes
                - "mode_log_probs": [batch_size] log probs
                - "mode_probs": [batch_size, num_modes] probabilities
        """
        modes, log_probs = self.mode_policy.sample_mode(
            state_embedding,
            deterministic=deterministic
        )
        probs = self.mode_policy.forward(state_embedding)

        return {
            "modes": modes,
            "mode_log_probs": log_probs,
            "mode_probs": probs,
        }


class PPOModePolicy:
    """
    PPO objective for mode policy optimization.

    Implements the PPO-style clipped objective for the mode policy:
    L_mode-RL = -E_t[min(r_t^φ A_t, clip(r_t^φ, 1-ε, 1+ε) A_t)]
    """

    @staticmethod
    def compute_loss(
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PPO loss for mode policy.

        Args:
            current_log_probs: [batch_size] log probs from current policy
            old_log_probs: [batch_size] log probs from old policy
            advantages: [batch_size] advantages
            clip_ratio: PPO clipping range (epsilon)
            entropy_coef: Entropy regularization coefficient

        Returns:
            loss: Scalar loss
            metrics: Dictionary with loss components
        """
        # Importance sampling ratio
        ratio = torch.exp(current_log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages

        policy_loss = -torch.mean(torch.min(surr1, surr2))

        metrics = {
            "policy_loss": policy_loss.item(),
            "mean_ratio": ratio.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }

        return policy_loss, metrics
