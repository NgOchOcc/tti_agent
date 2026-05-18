"""
CVI-Gated Self-Distillation for CVI-SDAR.

Implements:
1. CVI-gated distillation loss with reliability gating
2. Mode-level and token/action-level distillation
3. Uncertainty-aware gating mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class CVIGate(nn.Module):
    """
    Detached CVI gate for gated distillation.

    Prevents harmful teacher signals when:
    - CVI advantage is low (not worth imitating)
    - Teacher branches are uncertain
    - Advantage is below margin threshold

    Gate formula:
    g_t = sg[σ(β(Â_t^CVI(m_T*) - η*σ̂_t - ρ))]

    where:
    - sg: stop gradient (detached)
    - σ: sigmoid
    - Â_t^CVI: empirical CVI advantage
    - σ̂_t: uncertainty estimate
    - β: temperature scaling
    - η: uncertainty penalty weight
    - ρ: margin threshold
    """

    def __init__(
        self,
        beta: float = 2.0,
        eta: float = 0.5,
        rho: float = 0.0,
        use_sigmoid: bool = True,
    ):
        """
        Args:
            beta: Temperature scaling for sigmoid
            eta: Weight for uncertainty penalty
            rho: Margin threshold
            use_sigmoid: If True, apply sigmoid; if False, use ReLU
        """
        super().__init__()
        self.beta = beta
        self.eta = eta
        self.rho = rho
        self.use_sigmoid = use_sigmoid

    def forward(
        self,
        cvi_advantage: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the CVI gate.

        Args:
            cvi_advantage: [batch_size] empirical CVI advantages
            uncertainty: [batch_size] uncertainty estimates (default=0)

        Returns:
            gates: [batch_size] gate values in [0, 1]
        """
        if uncertainty is None:
            uncertainty = torch.zeros_like(cvi_advantage)

        # Compute gate input: Â_CVI - η*σ̂ - ρ
        gate_input = cvi_advantage - self.eta * uncertainty - self.rho

        # Apply temperature scaling
        gate_input = self.beta * gate_input

        # Apply nonlinearity
        if self.use_sigmoid:
            gates = torch.sigmoid(gate_input)
        else:
            gates = F.relu(gate_input)

        # Stop gradients through the gate
        gates = gates.detach()

        return gates


class CVIDistillationLoss(nn.Module):
    """
    CVI-gated self-distillation loss for CVI-SDAR training.

    Combines:
    1. Mode-level distillation: Encourages imitating teacher modes
    2. Token/action-level distillation: Encourages imitating teacher actions
    3. CVI gating: Only distill when advantage is positive and certain
    """

    def __init__(
        self,
        num_modes: int = 6,
        beta: float = 2.0,
        eta: float = 0.5,
        rho: float = 0.0,
        token_weight: float = 0.5,
        use_sigmoid_gate: bool = True,
    ):
        """
        Args:
            num_modes: Number of modes
            beta: Gate temperature
            eta: Uncertainty penalty weight
            rho: Margin threshold
            token_weight: Weight for token-level distillation vs mode-level
            use_sigmoid_gate: If True, use sigmoid for gate; if False, use ReLU
        """
        super().__init__()
        self.num_modes = num_modes
        self.token_weight = token_weight

        # CVI gate
        self.cvi_gate = CVIGate(
            beta=beta,
            eta=eta,
            rho=rho,
            use_sigmoid=use_sigmoid_gate
        )

    def forward(
        self,
        # Mode policy outputs
        mode_logits: torch.Tensor,  # [batch_size, num_modes]
        # Teacher signal
        teacher_modes: torch.Tensor,  # [batch_size] teacher mode indices
        # CVI signal
        cvi_advantages: torch.Tensor,  # [batch_size] empirical CVI advantages
        uncertainty: Optional[torch.Tensor] = None,  # [batch_size]
        # Action outputs (optional)
        action_logits: Optional[torch.Tensor] = None,  # [batch_size*seq_len, vocab_size]
        teacher_actions: Optional[torch.Tensor] = None,  # [batch_size*seq_len]
        action_masks: Optional[torch.Tensor] = None,  # [batch_size*seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gated self-distillation loss.

        Args:
            mode_logits: Raw logits from mode policy
            teacher_modes: Teacher mode indices (1-hot targets)
            cvi_advantages: Empirical CVI advantages (U_m_T* - U_ANSWER)
            uncertainty: Uncertainty across branch samples (optional)
            action_logits: Raw logits from action policy (optional)
            teacher_actions: Teacher action indices (optional)
            action_masks: Mask for valid positions (optional)

        Returns:
            Dictionary with:
                - "mode_sd_loss": Mode-level distillation loss
                - "token_sd_loss": Token-level distillation loss
                - "total_loss": Weighted combination
                - "gate": Gate values for monitoring
                - "metrics": Dictionary of metrics
        """
        batch_size = mode_logits.shape[0]

        # Compute CVI gate
        gates = self.cvi_gate(cvi_advantages, uncertainty)  # [batch_size]

        # Mode-level distillation loss
        # L_mode-SD = -E_t[g_t * log π_φ(m_T* | h_t, b_t)]
        mode_log_probs = F.log_softmax(mode_logits, dim=-1)  # [batch_size, num_modes]

        # Get log probs of teacher modes
        teacher_log_probs = mode_log_probs[
            torch.arange(batch_size), teacher_modes
        ]  # [batch_size]

        # Apply gate to distillation
        gated_mode_loss = -gates * teacher_log_probs
        mode_sd_loss = torch.mean(gated_mode_loss)

        # Token/action-level distillation (optional)
        token_sd_loss = torch.tensor(0.0, device=mode_logits.device)

        if action_logits is not None and teacher_actions is not None:
            token_sd_loss = self._compute_token_loss(
                action_logits, teacher_actions, gates, action_masks
            )

        # Weighted combination
        total_loss = (
            mode_sd_loss +
            self.token_weight * token_sd_loss
        )

        metrics = {
            "gate_mean": gates.mean().item(),
            "gate_std": gates.std().item(),
            "gate_min": gates.min().item(),
            "gate_max": gates.max().item(),
            "mode_sd_loss": mode_sd_loss.item(),
            "token_sd_loss": token_sd_loss.item() if isinstance(token_sd_loss, torch.Tensor) else 0.0,
            "cvi_advantage_mean": cvi_advantages.mean().item(),
            "cvi_advantage_std": cvi_advantages.std().item(),
            "uncertainty_mean": uncertainty.mean().item() if uncertainty is not None else 0.0,
        }

        return {
            "mode_sd_loss": mode_sd_loss,
            "token_sd_loss": token_sd_loss,
            "total_loss": total_loss,
            "gate": gates,
            "metrics": metrics,
        }

    def _compute_token_loss(
        self,
        action_logits: torch.Tensor,
        teacher_actions: torch.Tensor,
        gates: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute token/action-level distillation loss.

        L_tok-SD = -E_{t,i}[g_{t,i} * log π_θ(a_T_{t,i} | ...)]

        Args:
            action_logits: [total_tokens, vocab_size] or [batch_size, seq_len, vocab_size]
            teacher_actions: [total_tokens] or [batch_size, seq_len]
            gates: [batch_size] gate values
            action_masks: [total_tokens] or [batch_size, seq_len]

        Returns:
            Scalar loss
        """
        # Flatten if needed
        if action_logits.dim() == 3:
            batch_size, seq_len = action_logits.shape[0], action_logits.shape[1]

            # Expand gates to match token dimension
            gates_expanded = gates.unsqueeze(1).expand(batch_size, seq_len)
            gates_expanded = gates_expanded.reshape(-1)  # [batch_size * seq_len]

            action_logits = action_logits.reshape(-1, action_logits.shape[-1])
            teacher_actions = teacher_actions.reshape(-1)

        else:
            # Expand gates if needed
            if gates.shape[0] != action_logits.shape[0]:
                # Repeat gates for each token in sequence
                gates_expanded = gates.repeat_interleave(
                    action_logits.shape[0] // gates.shape[0]
                )
            else:
                gates_expanded = gates

        # Compute log probabilities
        action_log_probs = F.log_softmax(action_logits, dim=-1)  # [total_tokens, vocab_size]
        teacher_log_probs = action_log_probs[
            torch.arange(action_log_probs.shape[0]),
            teacher_actions
        ]  # [total_tokens]

        # Apply gates and mask
        gated_loss = -gates_expanded * teacher_log_probs

        if action_masks is not None:
            gated_loss = gated_loss * action_masks

            loss = gated_loss.sum() / action_masks.sum().clamp(min=1)
        else:
            loss = gated_loss.mean()

        return loss


class DistillationGateProfiling:
    """
    Utilities for profiling and analyzing the distillation gate behavior.
    """

    @staticmethod
    def analyze_gates(
        gates: np.ndarray,
        cvi_advantages: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Analyze gate statistics and effectiveness.

        Args:
            gates: [batch_size] gate values
            cvi_advantages: [batch_size] CVI advantages
            uncertainty: [batch_size] uncertainty values

        Returns:
            Dictionary of statistics
        """
        stats = {
            "gate_mean": float(np.mean(gates)),
            "gate_std": float(np.std(gates)),
            "gate_min": float(np.min(gates)),
            "gate_max": float(np.max(gates)),
            "gate_median": float(np.median(gates)),
            "num_gates_active": int(np.sum(gates > 0.5)),
            "pct_gates_active": 100.0 * np.sum(gates > 0.5) / len(gates),
        }

        # Correlation with CVI advantage
        if len(cvi_advantages) > 1:
            corr = float(np.corrcoef(gates, cvi_advantages)[0, 1])
            stats["gate_cvi_correlation"] = corr

        # Correlation with uncertainty if provided
        if uncertainty is not None and len(uncertainty) > 1:
            corr = float(np.corrcoef(gates, uncertainty)[0, 1])
            stats["gate_uncertainty_correlation"] = corr

        return stats

    @staticmethod
    def compute_distillation_efficiency(
        gates: np.ndarray,
        cvi_advantages: np.ndarray,
        mode_losses: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute distillation efficiency metrics.

        Args:
            gates: [batch_size] gate values
            cvi_advantages: [batch_size] CVI advantages
            mode_losses: [batch_size] mode losses

        Returns:
            Dictionary of efficiency metrics
        """
        # Gated vs ungated losses
        gated_losses = gates * mode_losses
        ungated_losses = mode_losses

        metrics = {
            "mean_gated_loss": float(np.mean(gated_losses)),
            "mean_ungated_loss": float(np.mean(ungated_losses)),
            "loss_reduction_ratio": 1.0 - (np.mean(gated_losses) / np.mean(ungated_losses)),

            # How well gate matches CVI signal
            "advantage_matched_gates": float(
                np.mean(gates[cvi_advantages > 0])
            ) if np.any(cvi_advantages > 0) else 0.0,
            "advantage_mismatched_gates": float(
                np.mean(gates[cvi_advantages <= 0])
            ) if np.any(cvi_advantages <= 0) else 0.0,
        }

        return metrics
