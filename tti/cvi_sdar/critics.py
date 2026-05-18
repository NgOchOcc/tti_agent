"""
Critic networks for CVI-SDAR.

Implements three value functions:
- V_ψ(h_t, b_t): Overall expected cost-aware utility
- Q_ψ(h_t, b_t, m): Expected utility of choosing mode m
- V_ans,ψ(h_t, b_t): Expected utility of answering immediately
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np


class CriticNetwork(nn.Module):
    """Lightweight critic network for value estimation.

    Takes input from agent's history/state encoding and produces scalar value estimates.
    Architecture: MLPwith configurable hidden dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        output_dim: int = 1,
        name: str = "critic"
    ):
        """
        Args:
            input_dim: Dimension of input (history embedding size)
            hidden_dims: Dimensions of hidden layers
            dropout: Dropout rate
            output_dim: Output dimension (usually 1 for scalar value)
            name: Name of the critic (for logging)
        """
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embedding: [batch_size, input_dim] or [batch_size, seq_len, input_dim]

        Returns:
            values: [batch_size] or [batch_size, seq_len] scalar values
        """
        original_shape = state_embedding.shape

        # Handle sequence inputs
        if len(state_embedding.shape) == 3:
            batch_size, seq_len, dim = state_embedding.shape
            state_embedding = state_embedding.reshape(-1, dim)

        values = self.network(state_embedding)  # [batch*seq, 1]

        # Reshape back if necessary
        if len(original_shape) == 3:
            batch_size, seq_len = original_shape[0], original_shape[1]
            values = values.reshape(batch_size, seq_len)
        else:
            values = values.squeeze(-1)

        return values


class MultiHeadCriticNetwork(nn.Module):
    """Critic network with multiple heads for different value types.

    Shared encoder + separate heads for efficiency.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: Tuple[int, ...] = (256,),
        head_hidden_dims: Tuple[int, ...] = (128,),
        dropout: float = 0.1,
        num_modes: int = 6,  # THINK, OBSERVE, ACT, VERIFY, RECOVER, ANSWER
    ):
        """
        Args:
            input_dim: Dimension of input
            encoder_hidden_dims: Hidden dimensions for shared encoder
            head_hidden_dims: Hidden dimensions for each head
            dropout: Dropout rate
            num_modes: Number of modes for Q-function
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_modes = num_modes

        # Shared encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()
        encoder_out_dim = prev_dim

        # Value head: V(h_t, b_t)
        v_layers = []
        prev_dim = encoder_out_dim
        for hidden_dim in head_hidden_dims:
            v_layers.append(nn.Linear(prev_dim, hidden_dim))
            v_layers.append(nn.ReLU())
            v_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        v_layers.append(nn.Linear(prev_dim, 1))
        self.v_head = nn.Sequential(*v_layers)

        # Answer value head: V_ans(h_t, b_t)
        ans_layers = []
        prev_dim = encoder_out_dim
        for hidden_dim in head_hidden_dims:
            ans_layers.append(nn.Linear(prev_dim, hidden_dim))
            ans_layers.append(nn.ReLU())
            ans_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        ans_layers.append(nn.Linear(prev_dim, 1))
        self.v_ans_head = nn.Sequential(*ans_layers)

        # Q-function head: Q(h_t, b_t, m) for each mode m
        q_layers = []
        prev_dim = encoder_out_dim
        for hidden_dim in head_hidden_dims:
            q_layers.append(nn.Linear(prev_dim, hidden_dim))
            q_layers.append(nn.ReLU())
            q_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        q_layers.append(nn.Linear(prev_dim, num_modes))
        self.q_head = nn.Sequential(*q_layers)

    def forward(
        self,
        state_embedding: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_embedding: [batch_size, input_dim]
            return_all: If True, return all values; if False, return only V and Q

        Returns:
            Dictionary with keys:
                - "v": [batch_size] value function estimates
                - "q": [batch_size, num_modes] Q-function estimates
                - "v_ans": [batch_size] answer value estimates (if return_all=True)
        """
        # Encode state
        encoded = self.encoder(state_embedding)

        # Get values
        v = self.v_head(encoded).squeeze(-1)  # [batch_size]
        q = self.q_head(encoded)  # [batch_size, num_modes]

        result = {
            "v": v,
            "q": q,
        }

        if return_all:
            v_ans = self.v_ans_head(encoded).squeeze(-1)  # [batch_size]
            result["v_ans"] = v_ans

        return result


def create_critics(
    input_dim: int,
    num_modes: int = 6,
    critic_hidden_dims: Tuple[int, ...] = (256, 128),
    dropout: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, CriticNetwork]:
    """
    Factory function to create the three critic networks.

    Args:
        input_dim: Input dimension (history embedding size)
        num_modes: Number of modes (for Q-function)
        critic_hidden_dims: Hidden layer dimensions for critics
        dropout: Dropout rate
        device: Device to place networks on

    Returns:
        Dictionary with keys:
            - "v": Overall value critic
            - "q": Q-function critic (outputs num_modes values)
            - "v_ans": Answer value critic
    """
    critics = {
        "v": CriticNetwork(
            input_dim=input_dim,
            hidden_dims=critic_hidden_dims,
            dropout=dropout,
            output_dim=1,
            name="V"
        ).to(device),
        "q": CriticNetwork(
            input_dim=input_dim,
            hidden_dims=critic_hidden_dims,
            dropout=dropout,
            output_dim=num_modes,  # One output per mode
            name="Q"
        ).to(device),
        "v_ans": CriticNetwork(
            input_dim=input_dim,
            hidden_dims=critic_hidden_dims,
            dropout=dropout,
            output_dim=1,
            name="V_ans"
        ).to(device),
    }

    return critics


class CriticLosses:
    """Computes critic losses for CVI-SDAR training."""

    @staticmethod
    def critic_loss(
        v_pred: torch.Tensor,
        q_pred: torch.Tensor,
        returns: torch.Tensor,
        modes: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute critic loss: MSE between predicted values and returns.

        Args:
            v_pred: [batch_size] predicted overall values
            q_pred: [batch_size, num_modes] predicted Q-values
            returns: [batch_size] target returns G_t
            modes: [batch_size] mode indices m_t (if provided, compute Q loss only for taken modes)
            reduction: "mean" or "sum"

        Returns:
            Scalar loss
        """
        v_loss = torch.mean((v_pred - returns) ** 2)

        if modes is not None:
            # Compute Q loss only for the mode that was taken
            q_targets = returns  # Same target as overall value
            q_taken = q_pred[torch.arange(len(modes)), modes]
            q_loss = torch.mean((q_taken - q_targets) ** 2)
        else:
            # Compute Q loss for all modes
            q_loss = torch.mean((q_pred - returns.unsqueeze(-1)) ** 2)

        total_loss = v_loss + q_loss

        return total_loss

    @staticmethod
    def answer_loss(
        v_ans_pred: torch.Tensor,
        answer_utilities: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute answer-now baseline loss.

        Args:
            v_ans_pred: [batch_size] predicted values for answering now
            answer_utilities: [batch_size] actual utilities from answering now
            reduction: "mean" or "sum"

        Returns:
            Scalar loss
        """
        loss = torch.mean((v_ans_pred - answer_utilities) ** 2)
        return loss
